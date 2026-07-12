use super::*;
pub(crate) use gam_problem::{LOG_STRENGTH_MAX, LOG_STRENGTH_MIN};
use gam_problem::{checked_exp_log_strength, checked_exp_log_strengths, validate_log_strength};

/// Closed numerical domain of every active flat log-strength coordinate.
///
/// These are real parameter-domain endpoints, not saturation points: callers
/// reject values outside the interval instead of clipping them onto a constant
/// objective plateau.

/// #1026 — how the per-atom ARD precisions are exposed to the OUTER PENALIZED QUASI-LAPLACE
/// optimizer.
///
/// The term's inner solve always reads a full per-atom, per-axis precision
/// table ([`SaeManifoldRho::log_ard`], a `Vec<Array1>` of length `K`); this
/// enum changes only how many DISTINCT outer hyperparameters the outer
/// optimizer searches over and how the flat outer vector reconstitutes that
/// table.
///
/// * [`ArdSharing::PerAtom`] — the historical default: every atom/axis ARD
///   strength is an independent outer coordinate, so the flat outer vector
///   carries `Σ_k d_k` ARD coordinates. Correct and selective for small K, but
///   the outer optimizer then faces `2 + Σ_k d_k` hyperparameters (≈ 32 770 at
///   K = 32 768 1-D atoms), each outer eval refitting the whole dictionary —
///   intractable at large K.
/// * [`ArdSharing::Shared`] — collapse the per-atom ARD to a handful of SHARED
///   strengths, one per axis index `j ∈ 0..max_d` (`max_d = max_k d_k`),
///   BROADCAST to every atom that owns axis `j`. The flat outer vector then
///   carries a constant `max_d` ARD coordinates (typically 1 or 2) regardless
///   of K, so the outer optimizer searches `sparse_dim + K + max_d`
///   hyperparameters. This is
///   a principled shared-λ tie: all atoms share one ARD precision per intrinsic
///   axis, exactly the standard "shared smoothing parameter across replicate
///   terms" REML reparameterization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArdSharing {
    /// One independent outer ARD coordinate per atom per axis (`Σ_k d_k`).
    PerAtom,
    /// One shared outer ARD coordinate per axis index, broadcast to all atoms.
    Shared,
}

#[cfg(test)]
mod log_strength_domain_tests {
    use super::*;
    use ndarray::array;

    fn fully_active_rho() -> SaeManifoldRho {
        SaeManifoldRho::new(0.0, 0.0, vec![array![0.0]]).with_log_lambda_block(vec![0.0])
    }

    #[test]
    fn every_active_flat_log_strength_has_one_closed_domain() {
        let rho = fully_active_rho();
        let dimension = rho.to_flat().len();
        for endpoint in [LOG_STRENGTH_MIN, LOG_STRENGTH_MAX] {
            let flat = Array1::from_elem(dimension, endpoint);
            let rebuilt = rho
                .from_flat(flat.view())
                .expect("both exact log-strength endpoints are in-domain");
            assert_eq!(rebuilt.to_flat(), flat);
        }
        assert_eq!(
            rho.flat_domain_lower_bound().unwrap(),
            Array1::from_elem(dimension, LOG_STRENGTH_MIN)
        );
        assert_eq!(
            rho.flat_domain_upper_bound().unwrap(),
            Array1::from_elem(dimension, LOG_STRENGTH_MAX)
        );

        for coordinate in 0..dimension {
            for invalid in [
                LOG_STRENGTH_MIN - 1.0,
                LOG_STRENGTH_MAX + 1.0,
                f64::NAN,
                f64::INFINITY,
            ] {
                let mut flat = Array1::zeros(dimension);
                flat[coordinate] = invalid;
                let error = rho
                    .from_flat(flat.view())
                    .expect_err("every emitted log strength must fail outside the domain");
                assert!(
                    error.contains("must be finite and in"),
                    "coordinate {coordinate}, value {invalid}: {error}"
                );
            }
        }
    }

    #[test]
    fn structurally_absent_sparse_placeholder_is_ignored_and_never_scaled() {
        let rho = SaeManifoldRho::new(17.0, 0.0, vec![Array1::<f64>::zeros(0)])
            .for_assignment(AssignmentMode::softmax(1.0));
        assert_eq!(rho.sparse_flat_index(), None);
        let mut irrelevant_placeholder = rho.clone();
        irrelevant_placeholder.log_lambda_sparse = f64::INFINITY;
        irrelevant_placeholder
            .validate_log_strength_domain()
            .expect("a non-coordinate placeholder is outside the objective domain");
        assert_eq!(rho.to_flat(), array![0.0]);

        let scaled = rho
            .seed_scaled_by_dispersion_for_assignment(1.0e300, AssignmentMode::softmax(1.0))
            .expect("dispersion scaling must not touch an absent sparse coordinate");
        assert_eq!(scaled.log_lambda_sparse, 17.0);
        assert_eq!(scaled.to_flat().len(), 1);
        scaled
            .validate_log_strength_domain()
            .expect("the active smooth coordinate remains valid");
    }
}

/// Whether assignment strength contributes an outer penalized quasi-Laplace coordinate.
///
/// The stored [`SaeManifoldRho::log_lambda_sparse`] value remains available to
/// the inner assignment prior, but the flat outer layout includes it only when
/// the assignment family has a non-constant strength-dependent objective:
///
/// * [`Self::PenaltyWeight`] always carries the coordinate (ordered Beta--Bernoulli and
///   threshold-gate priors).
/// * [`Self::SoftmaxEntropy`] carries it only for `K > 1`. At `K = 1` the
///   simplex assignment is identically one and its entropy is identically zero,
///   so there is no parameter to optimize or certify.
/// * [`Self::FixedSupport`] never carries it. Hard TopK sparsity is the support
///   constraint itself and has no assignment-strength penalty.
///
/// Keeping this distinction in the typed rho layout prevents a structurally
/// absent parameter from surviving as a held optimizer coordinate with a
/// nonzero, uncertifiable gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentStrengthLayout {
    PenaltyWeight,
    SoftmaxEntropy,
    FixedSupport,
}

impl AssignmentStrengthLayout {
    fn has_outer_coordinate(self, k_atoms: usize) -> bool {
        match self {
            Self::PenaltyWeight => true,
            Self::SoftmaxEntropy => k_atoms > 1,
            Self::FixedSupport => false,
        }
    }
}

/// REML-selected continuous hyperparameters for SAE-manifold.
#[derive(Debug, Clone)]
pub struct SaeManifoldRho {
    /// `log(lambda_sparse)` for softmax entropy or ThresholdGate gated L1, or the
    /// learnable `log(alpha)` offset for ordered Beta--Bernoulli assignment.
    pub log_lambda_sparse: f64,
    /// Typed assignment-strength layout. This is assignment-family state, not
    /// an optimizer mask: when the coordinate is structurally absent it is not
    /// emitted by [`Self::to_flat`] and cannot appear in the objective gradient.
    pub assignment_strength_layout: AssignmentStrengthLayout,
    /// Per-atom `log(lambda_smooth)` — one independent decoder-smoothness
    /// strength per atom `k` (length `K`, atom order). Atom `k`'s bending
    /// penalty `S_k` is scaled by `lambda_smooth[k] = exp(log_lambda_smooth[k])`,
    /// so distinct atoms can carry distinct smoothness strengths (#1556). Linear
    /// atoms have a null `S_k`, so their per-atom entry is a harmless no-op.
    ///
    /// Historically this was a single global scalar shared by every atom; the
    /// ergonomic [`SaeManifoldRho::new`] still accepts a scalar and BROADCASTS it
    /// to all `K` atoms (so the common "one global λ_smooth" call sites are
    /// unchanged), while [`SaeManifoldRho::with_per_atom_smooth`] sets a genuinely
    /// per-atom vector. The EFS / Fellner–Schall multiplicative update is already
    /// per-coordinate and writes each atom's entry independently.
    pub log_lambda_smooth: Vec<f64>,
    /// Per-atom, per-axis `log(alpha_kj)` ARD strengths. An empty per-atom
    /// block disables native coordinate ARD for that atom. The inner solve and
    /// every penalty assembler read THIS full table; `ard_sharing` only governs
    /// how the OUTER optimizer's flat coordinate vector maps onto it.
    pub log_ard: Vec<Array1<f64>>,
    /// #1026 — outer-optimizer ARD parameterization (per-atom vs shared). Does
    /// not change `log_ard`'s shape or the inner-solve math; only `to_flat` /
    /// `from_flat` consult it.
    pub ard_sharing: ArdSharing,
    /// #2231 §2a — per-output-block relevance weights `log(λ_ℓ)` for a manifold
    /// CROSSCODER, length `L-1` in stacked-block order (parallel to a term's
    /// [`crate::manifold::CrosscoderLayout::block_dims`]). EMPTY for a plain SAE
    /// (the historical case): the block sub-vector is APPENDED to the flat
    /// outer-coordinate layout AFTER the ARD block, so with an empty vector every
    /// existing consumer's cursor arithmetic (`to_flat` / `from_flat` /
    /// [`Self::ard_flat_index`]) is untouched and the plain-SAE flat vector is
    /// byte-identical.
    ///
    /// The block weight scales the augmented crosscoder target's block columns by
    /// `√λ_ℓ` (never the design), so it enters the criterion only through the
    /// per-block residual sum of squares and the `√λ_ℓ` target-scaling Jacobian —
    /// its closed-form REML variance ratio is
    /// [`crate::manifold::behavior::OutputBlock::reml_updated_log_lambda`] and its
    /// analytic outer gradient is
    /// [`crate::manifold::behavior::profiled_penalized_quasi_laplace_block_log_lambda_gradient`].
    pub log_lambda_block: Vec<f64>,
}

impl SaeManifoldRho {
    /// Build a ρ, BROADCASTING the single scalar `log_lambda_smooth` to all
    /// `K = log_ard.len()` atoms (#1556). The field is genuinely per-atom; this
    /// ergonomic constructor only seeds every atom with the same strength so the
    /// historical "one global λ_smooth" call sites need no change. Use
    /// [`Self::with_per_atom_smooth`] to seed distinct per-atom strengths.
    #[must_use]
    pub fn new(log_lambda_sparse: f64, log_lambda_smooth: f64, log_ard: Vec<Array1<f64>>) -> Self {
        let k = log_ard.len();
        Self {
            log_lambda_sparse,
            assignment_strength_layout: AssignmentStrengthLayout::PenaltyWeight,
            log_lambda_smooth: vec![log_lambda_smooth; k],
            log_ard,
            ard_sharing: ArdSharing::PerAtom,
            log_lambda_block: Vec::new(),
        }
    }

    /// Build a ρ with an explicit per-atom `log_lambda_smooth` vector (length
    /// `K`, atom order). Each atom `k`'s decoder-smoothness penalty `S_k` is then
    /// scaled by its own `exp(log_lambda_smooth[k])` (#1556).
    #[must_use]
    pub fn with_per_atom_smooth(
        log_lambda_sparse: f64,
        log_lambda_smooth: Vec<f64>,
        log_ard: Vec<Array1<f64>>,
    ) -> Self {
        Self {
            log_lambda_sparse,
            assignment_strength_layout: AssignmentStrengthLayout::PenaltyWeight,
            log_lambda_smooth,
            log_ard,
            ard_sharing: ArdSharing::PerAtom,
            log_lambda_block: Vec::new(),
        }
    }

    /// Build a ρ whose OUTER optimizer searches a constant `max_d` SHARED ARD
    /// strengths (one per axis index) instead of `Σ_k d_k` per-atom ones. The
    /// inner per-atom `log_ard` table is unchanged; the broadcast happens in
    /// `from_flat`. See [`ArdSharing`].
    #[must_use]
    pub fn new_shared_ard(
        log_lambda_sparse: f64,
        log_lambda_smooth: f64,
        log_ard: Vec<Array1<f64>>,
    ) -> Self {
        let k = log_ard.len();
        Self {
            log_lambda_sparse,
            assignment_strength_layout: AssignmentStrengthLayout::PenaltyWeight,
            log_lambda_smooth: vec![log_lambda_smooth; k],
            log_ard,
            ard_sharing: ArdSharing::Shared,
            log_lambda_block: Vec::new(),
        }
    }

    /// Return a copy of this ρ carrying the crosscoder per-block relevance weights
    /// `log(λ_ℓ)` (#2231 §2a). The block sub-vector is APPENDED to the flat
    /// layout after ARD; an empty `log_lambda_block` restores the plain-SAE
    /// byte-identical layout. The block order must match a term's
    /// [`crate::manifold::CrosscoderLayout::block_dims`].
    #[must_use]
    pub fn with_log_lambda_block(mut self, log_lambda_block: Vec<f64>) -> Self {
        self.log_lambda_block = log_lambda_block;
        self
    }

    /// Bind the flat assignment-strength layout to the term's assignment
    /// family. The `K = 1` Softmax case and every hard-TopK case are structural
    /// absences, not frozen coordinates.
    #[must_use]
    pub fn for_assignment(mut self, assignment_mode: AssignmentMode) -> Self {
        self.assignment_strength_layout = match assignment_mode {
            AssignmentMode::Softmax { .. } => AssignmentStrengthLayout::SoftmaxEntropy,
            AssignmentMode::TopK { .. } => AssignmentStrengthLayout::FixedSupport,
            AssignmentMode::OrderedBetaBernoulli { .. } | AssignmentMode::ThresholdGate { .. } => {
                AssignmentStrengthLayout::PenaltyWeight
            }
        };
        self
    }

    /// Assignment-strength layout bound to this rho.
    #[must_use]
    pub fn assignment_strength_layout(&self) -> AssignmentStrengthLayout {
        self.assignment_strength_layout
    }

    /// Flat index of `log_lambda_sparse`, or `None` when assignment strength is
    /// structurally absent from the outer problem.
    #[must_use]
    pub fn sparse_flat_index(&self) -> Option<usize> {
        self.assignment_strength_layout
            .has_outer_coordinate(self.k_atoms())
            .then_some(0)
    }

    /// First flat coordinate occupied by per-atom smoothness.
    #[must_use]
    pub fn smooth_flat_start(&self) -> usize {
        usize::from(self.sparse_flat_index().is_some())
    }

    /// Flat coordinate for atom `atom`'s smoothness strength.
    #[must_use]
    pub fn smooth_flat_index(&self, atom: usize) -> usize {
        assert!(
            atom < self.k_atoms(),
            "SaeManifoldRho::smooth_flat_index: atom {atom} outside K={}",
            self.k_atoms()
        );
        self.smooth_flat_start() + atom
    }

    /// Number of crosscoder output blocks `L-1` carried as outer coordinates
    /// (0 for a plain SAE).
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.log_lambda_block.len()
    }

    /// Largest per-atom ARD axis count `max_k d_k` (0 when ARD is disabled on
    /// every atom). This is the number of SHARED outer ARD coordinates in
    /// [`ArdSharing::Shared`] mode.
    #[must_use]
    pub fn max_ard_axes(&self) -> usize {
        self.log_ard.iter().map(|a| a.len()).max().unwrap_or(0)
    }

    /// The outer ARD parameterization ([`ArdSharing::PerAtom`] vs
    /// [`ArdSharing::Shared`]). Every derivative / trace / EFS / IFT-RHS
    /// consumer of the flat ρ layout must branch on this: the per-atom cursor
    /// walk `sparse_dim+K+Σ_{a<k} d_a + j` is only valid in
    /// [`ArdSharing::PerAtom`] mode;
    /// in [`ArdSharing::Shared`] mode atom `k`'s axis `j` maps onto the SINGLE
    /// shared coordinate `sparse_dim+K+j` (see [`Self::ard_flat_index`]), so several atoms
    /// alias one coordinate and their contributions must be ACCUMULATED — walking
    /// them as if per-atom both indexes OOB (flat len is only
    /// `sparse_dim+K+max_d`) and,
    /// when it does not panic, splits one shared strength across phantom slots.
    #[must_use]
    pub fn ard_sharing(&self) -> ArdSharing {
        self.ard_sharing
    }

    /// Flat outer-coordinate index that atom `k`'s ARD axis `j` writes to,
    /// consistent with [`Self::to_flat`] / [`Self::from_flat`].
    ///
    /// * [`ArdSharing::PerAtom`] — a UNIQUE coordinate per `(k, j)`:
    ///   `sparse_dim + K + Σ_{a<k} d_a + j`. Accumulating (`+=`) into it is therefore the
    ///   same as assigning.
    /// * [`ArdSharing::Shared`] — the shared per-axis coordinate
    ///   `sparse_dim + K + j`,
    ///   which EVERY atom owning axis `j` maps onto. Consumers MUST accumulate
    ///   (gradient / trace / RHS `+=`; EFS numerator/denominator summed over the
    ///   atoms owning the axis), since the outer optimizer searches one strength
    ///   per axis, broadcast to all sharing atoms — the chain rule
    ///   `∂/∂log α_j = Σ_{k owns j} ∂/∂log α_{kj}`.
    #[must_use]
    pub fn ard_flat_index(&self, atom: usize, axis: usize) -> usize {
        let k = self.log_lambda_smooth.len();
        let prefix = self.smooth_flat_start();
        match self.ard_sharing {
            ArdSharing::PerAtom => {
                let base: usize = self.log_ard[..atom].iter().map(|a| a.len()).sum();
                prefix + k + base + axis
            }
            ArdSharing::Shared => prefix + k + axis,
        }
    }

    /// Shift every scale-coupled penalty seed by the profiled reconstruction
    /// dispersion scale. SAE's Gaussian data-fit term is in squared output
    /// units, while `lambda_sparse`, `lambda_smooth`, and ARD precisions are
    /// absolute penalty weights; adding `log(phi_seed)` makes the seeded
    /// effective stiffness `lambda / phi_seed` dimensionless.
    pub fn seed_scaled_by_dispersion(&self, dispersion: f64) -> Result<Self, String> {
        self.seed_scaled_by_dispersion_with_sparse_policy(dispersion, true)
    }

    /// Assignment-aware seed scaling.
    ///
    /// The response-dispersion shift `λ → λ·φ_seed` makes the seeded effective
    /// stiffness `λ/φ_data` dimensionless — but that identity is derived from the
    /// Gaussian penalized-likelihood normal equations on a FIXED linear design.
    /// It is well-founded for the separable-gate modes (softmax entropy /
    /// ThresholdGate gated-L1), whose per-row gates are held at their seed weighting
    /// while the decoder/coordinates are refit, so `λ/φ` is exactly the effective
    /// stiffness.
    ///
    /// ordered Beta--Bernoulli is different in kind. Its per-row Bernoulli gates are FREE latent
    /// variables the inner joint solve co-optimizes with the coordinates and
    /// decoder. A response-dispersion-WEAKENED smoothness/ARD seed
    /// (`φ_seed ≪ 1` at any non-trivial noise scale) hands that extra gate +
    /// coordinate freedom enough slack to interpolate the noise: the inner solve
    /// overfits, the reconstruction dispersion `φ̂` collapses toward 0, and the
    /// Fellner–Schall multiplicative fixed point (`λ_new ∝ φ̂`) then spirals the
    /// smoothing/ARD penalties to zero — a degenerate outer basin the ρ-optimizer
    /// stalls in (#1744: ordered_beta_bernoulli n=40 σ=0.18 stalled at EV 0.86). The ordered Beta--Bernoulli sparse
    /// coordinate is additionally a dimensionless log-alpha concentration offset,
    /// not a squared-output-unit penalty weight, so it was never dispersion-
    /// scalable either. NONE of the ordered Beta--Bernoulli ρ coordinates therefore admit the
    /// Gaussian response-dispersion scaling; the seed stays at its absolute
    /// (already dimensionless) construction values, which keeps the smoothing/ARD
    /// penalties strong enough that the inner ordered Beta--Bernoulli solve cannot overfit at the seed
    /// and the EFS fixed point lands on the interior optimum instead of the
    /// zero-penalty collapse. The separable-gate modes are byte-for-byte
    /// unchanged.
    pub fn seed_scaled_by_dispersion_for_assignment(
        &self,
        dispersion: f64,
        assignment_mode: AssignmentMode,
    ) -> Result<Self, String> {
        let bound = self.clone().for_assignment(assignment_mode);
        if matches!(assignment_mode, AssignmentMode::OrderedBetaBernoulli { .. }) {
            // Validate the dispersion for parity with the scaled path (a
            // non-finite/​non-positive φ is still a caller error), then return the
            // unscaled seed: no ordered Beta--Bernoulli ρ coordinate is response-dispersion-scalable.
            if !(dispersion.is_finite() && dispersion > 0.0) {
                return Err(format!(
                    "SaeManifoldRho::seed_scaled_by_dispersion_for_assignment: dispersion must \
                     be finite and positive; got {dispersion}"
                ));
            }
            bound.validate_log_strength_domain()?;
            return Ok(bound);
        }
        // Separable-gate modes (softmax entropy / ThresholdGate gated-L1).
        //
        // #1782 — a SINGLE-atom (K = 1) fit has no cross-atom routing, so the
        // response-dispersion identity `λ/φ` is exactly the effective stiffness
        // and full scaling is well-founded: keep it BYTE-FOR-BYTE (this is the
        // regime the planted-circle noise-scale sweep pins). But a MULTI-atom
        // (K > 1) fit couples the per-atom decoders and coordinates through the
        // shared routing gate, and on clean data `φ_seed ≪ 1` the dispersion
        // shift `ln φ_seed` WEAKENS the decoder-smoothness / ARD seed toward
        // zero. That hands the coupled `(coords, decoders)` block enough slack to
        // overfit AT THE SEED, driving the undamped per-row / cross-row joint
        // Hessian indefinite — a non-PD seed whose quasi-Laplace score log-det is
        // undefined. Because the SAE fit runs a single seed (`max_seeds = 1`),
        // the EFS startup validation then rejects it with "no candidate seeds
        // passed outer startup validation" (the #1782 softmax / threshold-gate failure),
        // exactly where ordered_beta_bernoulli — which is never dispersion-weakened — survives.
        //
        // Fix: for K > 1 keep the seed decoder-smoothness / ARD from being
        // WEAKENED below their (dimensionless) construction strength — floor the
        // shift at 0 so noisy data (`φ > 1`) still STRENGTHENS smoothing (the
        // well-founded direction) while clean data can no longer collapse the
        // seed penalties into the non-PD basin. The sparse (gate) coordinate,
        // which does not enter the decoder Hessian, keeps its full dispersion
        // scaling. The EFS fixed point then descends each λ from this feasible,
        // PD seed to the same interior optimum.
        if bound.log_lambda_smooth.len() <= 1 {
            return bound.seed_scaled_by_dispersion_with_sparse_policy(dispersion, true);
        }
        if !(dispersion.is_finite() && dispersion > 0.0) {
            return Err(format!(
                "SaeManifoldRho::seed_scaled_by_dispersion_for_assignment: dispersion must \
                 be finite and positive; got {dispersion}"
            ));
        }
        let shift = dispersion.ln();
        let smooth_ard_shift = shift.max(0.0);
        let mut scaled = bound;
        if scaled.sparse_flat_index().is_some() {
            scaled.log_lambda_sparse += shift;
        }
        for value in &mut scaled.log_lambda_smooth {
            *value += smooth_ard_shift;
        }
        for atom in &mut scaled.log_ard {
            for value in atom.iter_mut() {
                *value += smooth_ard_shift;
            }
        }
        scaled.validate_log_strength_domain()?;
        Ok(scaled)
    }

    pub(crate) fn seed_scaled_by_dispersion_with_sparse_policy(
        &self,
        dispersion: f64,
        scale_sparse: bool,
    ) -> Result<Self, String> {
        if !(dispersion.is_finite() && dispersion > 0.0) {
            return Err(format!(
                "SaeManifoldRho::seed_scaled_by_dispersion: dispersion must be finite and \
                 positive; got {dispersion}"
            ));
        }
        let shift = dispersion.ln();
        let mut scaled = self.clone();
        if scale_sparse && scaled.sparse_flat_index().is_some() {
            scaled.log_lambda_sparse += shift;
        }
        for value in &mut scaled.log_lambda_smooth {
            *value += shift;
        }
        for atom in &mut scaled.log_ard {
            for value in atom.iter_mut() {
                *value += shift;
            }
        }
        scaled.validate_log_strength_domain()?;
        Ok(scaled)
    }

    pub fn lambda_sparse(&self) -> Result<f64, String> {
        checked_exp_log_strength(self.log_lambda_sparse)
            .map_err(|error| format!("assignment log strength: {error}"))
    }

    /// Number of atoms `K` carried by the per-atom smoothness vector.
    #[must_use]
    pub fn k_atoms(&self) -> usize {
        self.log_lambda_smooth.len()
    }

    /// Smoothness strength `exp(log_lambda_smooth[k])` for atom `k` (#1556).
    /// Computational entry points validate the rho domain before calling this
    /// exact, unsaturated map.
    #[must_use]
    pub fn lambda_smooth_for(&self, atom: usize) -> Result<f64, String> {
        let log_strength = self.log_lambda_smooth.get(atom).copied().ok_or_else(|| {
            format!(
                "smoothness atom {atom} is outside K={}",
                self.log_lambda_smooth.len()
            )
        })?;
        checked_exp_log_strength(log_strength)
            .map_err(|error| format!("smoothness log strength at atom {atom}: {error}"))
    }

    /// All `K` per-atom smoothness strengths `exp(log_lambda_smooth[k])`, atom
    /// order. Convenience for threading per-atom λ into the penalty assemblers
    /// (#1556).
    #[must_use]
    pub fn lambda_smooth_vec(&self) -> Result<Vec<f64>, String> {
        checked_exp_log_strengths(self.log_lambda_smooth.iter().copied())
            .map_err(|error| format!("smoothness log strength: {error}"))
    }

    /// Validate and materialize the complete per-atom ARD precision table once.
    ///
    /// ARD consumers call this before entering their row/atom kernels and reuse
    /// the returned physical precisions.  That gives value, gradient, Hessian,
    /// trace, and IFT channels the identical `alpha = exp(log_alpha)` map while
    /// avoiding a transcendental evaluation for every observation.  Validation
    /// is atomic: no table escapes unless every coordinate lies in the shared
    /// exact log-strength domain, and the first error is deterministic in
    /// `(atom, axis)` order.
    pub(crate) fn ard_precisions(&self) -> Result<Vec<Array1<f64>>, String> {
        let mut precisions = Vec::with_capacity(self.log_ard.len());
        for (atom, log_block) in self.log_ard.iter().enumerate() {
            let mut block = Array1::<f64>::zeros(log_block.len());
            for (axis, (&log_alpha, alpha)) in log_block.iter().zip(block.iter_mut()).enumerate() {
                *alpha = checked_exp_log_strength(log_alpha).map_err(|error| {
                    format!("ARD log precision at atom {atom}, axis {axis}: {error}")
                })?;
            }
            precisions.push(block);
        }
        Ok(precisions)
    }

    /// Validate every log-strength represented in the flat outer layout against
    /// the supported closed domain. A structurally absent assignment strength is
    /// deliberately ignored: it is not an objective coordinate and its stored
    /// placeholder cannot affect the corresponding assignment family.
    pub(crate) fn validate_log_strength_domain(&self) -> Result<(), String> {
        if self.sparse_flat_index().is_some()
            && validate_log_strength(self.log_lambda_sparse).is_err()
        {
            return Err(format!(
                "assignment log strength must be finite and in [{LOG_STRENGTH_MIN}, \
                 {LOG_STRENGTH_MAX}]; got {}",
                self.log_lambda_sparse
            ));
        }
        for (atom, &value) in self.log_lambda_smooth.iter().enumerate() {
            if validate_log_strength(value).is_err() {
                return Err(format!(
                    "smoothness log strength at atom {atom} must be finite and in \
                     [{LOG_STRENGTH_MIN}, {LOG_STRENGTH_MAX}]; got {value}"
                ));
            }
        }
        for (atom, block) in self.log_ard.iter().enumerate() {
            for (axis, &value) in block.iter().enumerate() {
                if validate_log_strength(value).is_err() {
                    return Err(format!(
                        "ARD log precision at atom {atom}, axis {axis} must be finite and in \
                         [{LOG_STRENGTH_MIN}, {LOG_STRENGTH_MAX}]; got {value}"
                    ));
                }
            }
        }
        for (block, &value) in self.log_lambda_block.iter().enumerate() {
            if validate_log_strength(value).is_err() {
                return Err(format!(
                    "block log strength at block {block} must be finite and in \
                     [{LOG_STRENGTH_MIN}, {LOG_STRENGTH_MAX}]; got {value}"
                ));
            }
        }
        Ok(())
    }

    /// Objective-domain lower face in flat-rho layout. Every emitted coordinate
    /// is a log strength, so all coordinates share the same exact endpoint.
    pub(crate) fn flat_domain_lower_bound(&self) -> Option<Array1<f64>> {
        let len = self.to_flat().len();
        if len == 0 {
            return None;
        }
        Some(Array1::from_elem(len, LOG_STRENGTH_MIN))
    }

    /// Objective-domain upper face in flat-rho layout; see
    /// [`Self::flat_domain_lower_bound`].
    pub(crate) fn flat_domain_upper_bound(&self) -> Option<Array1<f64>> {
        let len = self.to_flat().len();
        if len == 0 {
            return None;
        }
        Some(Array1::from_elem(len, LOG_STRENGTH_MAX))
    }

    /// Flatten ρ into the contiguous outer-coordinate vector the generic
    /// `OuterObjective` engine optimises over.
    ///
    /// Layout: `[<optional sparse>, <K smooth>, <ARD>, <L-1 block>]`, where
    /// `<optional sparse>` contains `log_lambda_sparse` exactly when
    /// [`Self::sparse_flat_index`] is `Some`, and is otherwise empty. The
    /// `<K smooth>` is the per-atom `log_lambda_smooth[k]` in atom order
    /// (`k in 0..K`), so the smoothness block carries `K` outer coordinates, not 1
    /// (#1556). The trailing `<L-1 block>` is the crosscoder per-block
    /// `log_lambda_block[ℓ]` (#2231 §2a), APPENDED after ARD and EMPTY for a plain
    /// SAE (so the plain-SAE flat vector is byte-identical).
    ///
    /// * [`ArdSharing::PerAtom`] — the `<ARD>` block concatenates each atom
    ///   `k`'s per-axis `log_ard[k][j]` in atom order, axis `j` in `0..d_k`.
    ///   Empty per-atom blocks contribute no outer coordinates, so the length is
    ///   `sparse_dim + K + Σ_k d_k`.
    /// * [`ArdSharing::Shared`] — the `<ARD>` block is a constant `max_d =
    ///   max_k d_k` SHARED strengths, one per axis index `j`. Each shared value
    ///   is the mean of `log_ard[k][j]` over the atoms that own axis `j` (an
    ///   exact read-back when the table is already broadcast, which it always is
    ///   under this mode); the length is `sparse_dim + K + max_d` regardless of d.
    ///   (Smoothness stays per-atom in both modes; `ard_sharing` governs ARD
    ///   only.)
    ///
    /// [`Self::from_flat`] is the exact inverse and reads the same layout from
    /// `self` (its `log_ard` shape + `ard_sharing`).
    pub fn to_flat(&self) -> Array1<f64> {
        let smooth_start = self.smooth_flat_start();
        match self.ard_sharing {
            ArdSharing::PerAtom => {
                let k = self.log_lambda_smooth.len();
                let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
                let block_len = self.log_lambda_block.len();
                let mut out = Array1::<f64>::zeros(smooth_start + k + ard_len + block_len);
                if let Some(index) = self.sparse_flat_index() {
                    out[index] = self.log_lambda_sparse;
                }
                for (atom, &v) in self.log_lambda_smooth.iter().enumerate() {
                    out[smooth_start + atom] = v;
                }
                let mut cursor = smooth_start + k;
                for axis in &self.log_ard {
                    for &v in axis.iter() {
                        out[cursor] = v;
                        cursor += 1;
                    }
                }
                // #2231 §2a — the crosscoder block weights are APPENDED after ARD
                // (empty ⇒ byte-identical plain-SAE layout).
                for &v in &self.log_lambda_block {
                    out[cursor] = v;
                    cursor += 1;
                }
                out
            }
            ArdSharing::Shared => {
                let k = self.log_lambda_smooth.len();
                let max_d = self.max_ard_axes();
                let block_len = self.log_lambda_block.len();
                let mut out = Array1::<f64>::zeros(smooth_start + k + max_d + block_len);
                if let Some(index) = self.sparse_flat_index() {
                    out[index] = self.log_lambda_sparse;
                }
                for (atom, &v) in self.log_lambda_smooth.iter().enumerate() {
                    out[smooth_start + atom] = v;
                }
                // Per-axis shared value = mean over atoms owning that axis. The
                // table is broadcast (all owners equal) under this mode, so the
                // mean is an exact read-back; averaging is only a defensive
                // collapse if an externally-built table is non-uniform.
                for j in 0..max_d {
                    let mut acc = 0.0;
                    let mut count = 0usize;
                    for atom in &self.log_ard {
                        if j < atom.len() {
                            acc += atom[j];
                            count += 1;
                        }
                    }
                    out[smooth_start + k + j] = if count > 0 { acc / count as f64 } else { 0.0 };
                }
                // #2231 §2a — crosscoder block weights appended after the shared
                // ARD block (empty ⇒ byte-identical).
                for (b, &v) in self.log_lambda_block.iter().enumerate() {
                    out[smooth_start + k + max_d + b] = v;
                }
                out
            }
        }
    }

    /// Rebuild a ρ with this ρ's per-atom ARD dimensions from a flat
    /// outer-coordinate vector produced by [`Self::to_flat`].
    ///
    /// The per-atom dims (and ARD sharing mode) are taken from `&self` (the ARD
    /// layout is a fixed property of the term shape; the engine only moves the
    /// values). The flat vector must have length
    /// `sparse_dim + K + Σ_k len(log_ard[k])` in
    /// [`ArdSharing::PerAtom`] mode, or `sparse_dim + K + max_k d_k` in
    /// [`ArdSharing::Shared`] mode, where `K = len(log_lambda_smooth)` carries the
    /// per-atom smoothness coordinates (#1556) and the few shared per-axis ARD
    /// values are BROADCAST back to every atom that owns that axis, rebuilding the
    /// full per-atom table the inner solve consumes.
    pub fn from_flat(&self, flat: ArrayView1<'_, f64>) -> Result<SaeManifoldRho, String> {
        let smooth_start = self.smooth_flat_start();
        let rebuilt = match self.ard_sharing {
            ArdSharing::PerAtom => {
                let k = self.log_lambda_smooth.len();
                let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
                let block_len = self.log_lambda_block.len();
                let expected = smooth_start + k + ard_len + block_len;
                if flat.len() != expected {
                    return Err(format!(
                        "SaeManifoldRho::from_flat: flat length {} != sparse_dim + K + \
                         Σ d_k + (L-1) = {expected}",
                        flat.len()
                    ));
                }
                let log_lambda_smooth: Vec<f64> =
                    (0..k).map(|atom| flat[smooth_start + atom]).collect();
                let mut log_ard = Vec::with_capacity(self.log_ard.len());
                let mut cursor = smooth_start + k;
                for axis in &self.log_ard {
                    let d = axis.len();
                    let mut block = Array1::<f64>::zeros(d);
                    for (j, slot) in block.iter_mut().enumerate() {
                        *slot = flat[cursor + j];
                    }
                    cursor += d;
                    log_ard.push(block);
                }
                // #2231 §2a — the appended crosscoder block tail (empty ⇒ no-op).
                let log_lambda_block: Vec<f64> = (0..block_len).map(|b| flat[cursor + b]).collect();
                SaeManifoldRho {
                    log_lambda_sparse: self
                        .sparse_flat_index()
                        .map_or(self.log_lambda_sparse, |index| flat[index]),
                    assignment_strength_layout: self.assignment_strength_layout,
                    log_lambda_smooth,
                    log_ard,
                    ard_sharing: ArdSharing::PerAtom,
                    log_lambda_block,
                }
            }
            ArdSharing::Shared => {
                let k = self.log_lambda_smooth.len();
                let max_d = self.max_ard_axes();
                let block_len = self.log_lambda_block.len();
                let expected = smooth_start + k + max_d + block_len;
                if flat.len() != expected {
                    return Err(format!(
                        "SaeManifoldRho::from_flat: shared-ARD flat length {} != sparse_dim + K + \
                         max_d + (L-1) = {expected}",
                        flat.len()
                    ));
                }
                let log_lambda_smooth: Vec<f64> =
                    (0..k).map(|atom| flat[smooth_start + atom]).collect();
                // Broadcast the shared per-axis strengths into each atom's block,
                // preserving every atom's own `d_k` (a `d_k`-axis atom reads the
                // first `d_k` shared values). This rebuilds the full per-atom
                // table the construction / penalty assemblers read unchanged.
                let mut log_ard = Vec::with_capacity(self.log_ard.len());
                for axis in &self.log_ard {
                    let d = axis.len();
                    let mut block = Array1::<f64>::zeros(d);
                    for (j, slot) in block.iter_mut().enumerate() {
                        *slot = flat[smooth_start + k + j];
                    }
                    log_ard.push(block);
                }
                // #2231 §2a — the appended crosscoder block tail (empty ⇒ no-op).
                let log_lambda_block: Vec<f64> = (0..block_len)
                    .map(|b| flat[smooth_start + k + max_d + b])
                    .collect();
                SaeManifoldRho {
                    log_lambda_sparse: self
                        .sparse_flat_index()
                        .map_or(self.log_lambda_sparse, |index| flat[index]),
                    assignment_strength_layout: self.assignment_strength_layout,
                    log_lambda_smooth,
                    log_ard,
                    ard_sharing: ArdSharing::Shared,
                    log_lambda_block,
                }
            }
        };
        rebuilt.validate_log_strength_domain()?;
        Ok(rebuilt)
    }
}
