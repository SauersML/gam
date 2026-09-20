use super::*;
pub(crate) use gam_problem::{LOG_STRENGTH_MAX, LOG_STRENGTH_MIN};
use gam_problem::{checked_exp_log_strength, checked_exp_log_strengths, validate_log_strength};

#[cfg(test)]
mod log_strength_domain_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn structurally_absent_sparse_placeholder_is_ignored_and_never_scaled() {
        let softmax = SaeAssignment::from_blocks_with_mode_and_manifolds(
            ndarray::Array2::<f64>::zeros((1, 1)),
            vec![ndarray::Array2::<f64>::zeros((1, 1))],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .expect("one logit column, coordinate block and manifold");
        let rho =
            SaeManifoldRho::new(17.0, 0.0, vec![Array1::<f64>::zeros(0)]).for_assignment(&softmax);
        assert_eq!(rho.sparse_flat_index(), None);
        let mut irrelevant_placeholder = rho.clone();
        irrelevant_placeholder.log_lambda_sparse = f64::INFINITY;
        irrelevant_placeholder
            .validate_log_strength_domain()
            .expect("a non-coordinate placeholder is outside the objective domain");
        assert_eq!(rho.flat_coordinates(), array![0.0]);

        let scaled = rho
            .seed_scaled_by_dispersion_for_assignment(1.0e300, &softmax)
            .expect("dispersion scaling must not touch an absent sparse coordinate");
        assert_eq!(scaled.log_lambda_sparse, 17.0);
        assert_eq!(scaled.flat_coordinates().len(), 1);
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
/// * [`Self::PenaltyWeight`] always carries the coordinate (threshold-gate prior).
/// * [`Self::ConcentrationOffset`] always carries it as `log(α/α_mode)` (ordered
///   Beta--Bernoulli with an effectively learnable concentration).
/// * [`Self::FixedConcentration`] never carries it. With the concentration fixed the
///   ordered Beta--Bernoulli prior is complete at weight one, with its constant partition
///   `Σ_k log C(a_k, N)`, so nothing optimized enters it (#2933 F45).
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
    ConcentrationOffset,
    FixedConcentration,
    SoftmaxEntropy,
    FixedSupport,
}

impl AssignmentStrengthLayout {
    /// The layout `assignment`'s family and effective concentration predicate call for.
    pub(crate) fn of(assignment: &SaeAssignment) -> Self {
        match assignment.mode {
            AssignmentMode::Softmax { .. } => Self::SoftmaxEntropy,
            AssignmentMode::TopK { .. } => Self::FixedSupport,
            AssignmentMode::ThresholdGate { .. } => Self::PenaltyWeight,
            AssignmentMode::OrderedBetaBernoulli { .. } => {
                if assignment.effective_alpha_is_learnable() {
                    Self::ConcentrationOffset
                } else {
                    Self::FixedConcentration
                }
            }
        }
    }

    fn has_outer_coordinate(self, k_atoms: usize) -> bool {
        match self {
            Self::PenaltyWeight | Self::ConcentrationOffset => true,
            Self::SoftmaxEntropy => k_atoms > 1,
            Self::FixedConcentration | Self::FixedSupport => false,
        }
    }
}

/// REML-selected continuous hyperparameters for SAE-manifold.
#[derive(Debug, Clone)]
pub struct SaeManifoldRho {
    /// `log(lambda_sparse)` for softmax entropy or ThresholdGate gated L1. For ordered
    /// Beta--Bernoulli it is the concentration offset `log(α/α_mode)` while α is
    /// effectively learnable, and an inert placeholder the prior never reads otherwise; see
    /// `SaeAssignment::ordered_beta_bernoulli_prior_parameters`.
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
    /// Per-atom, per-axis `log(alpha_kj)` ARD strengths. Every coordinate atom
    /// carries a full `log_ard` block, one entry per latent axis: the coordinate
    /// prior is what makes each row's coordinate posterior proper, so an empty
    /// block is refused at criterion entry (`validated_ard_precisions`). Each
    /// `(k, j)` entry is its own outer coordinate at every `K` (#3824): the
    /// prior family and the REML/LAML criterion do not change with the atom
    /// count.
    pub log_ard: Vec<Array1<f64>>,
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
    /// per-block residual sum of squares and the `√λ_ℓ` target-scaling Jacobian.
    pub log_lambda_block: Vec<f64>,
    /// #2604 — per-atom sectional curvature `kappa` for constant-curvature
    /// atoms, in atom order. EMPTY for every dictionary without one, which is
    /// the historical case: the curvature sub-vector is APPENDED to the flat
    /// layout AFTER the block tail, so an empty vector leaves every existing
    /// cursor arithmetic and the plain-SAE flat vector byte-identical — the
    /// same discipline `log_lambda_block` follows.
    ///
    /// Carried RAW, not as `log kappa`, matching the constant-curvature
    /// convention in `gam-models`' spatial optimizer: the family
    /// `S^d <- R^d -> H^d` passes continuously through `kappa = 0`, so flat
    /// space must be an INTERIOR point of the coordinate. A log parameterisation
    /// would put it at an unreachable boundary and could never fit a flat atom,
    /// nor cross from spherical to hyperbolic.
    ///
    /// The criterion moves with `kappa` only through the atom's penalty Gram —
    /// a constant-curvature atom's basis is a monomial patch in the TANGENT
    /// coordinate and does not depend on `kappa` — so the whole channel is
    /// `gam_geometry::constant_curvature_dirichlet_penalty_kappa_derivative`.
    pub kappa: Vec<f64>,
    /// Atom index owned by each entry of [`Self::kappa`], in the same order.
    ///
    /// This mapping is structural, not another outer coordinate. A mixed
    /// dictionary must not emit dummy curvature coordinates for flat/periodic
    /// atoms: those coordinates have identically-zero gradients and make the
    /// outer Hessian singular. Keeping only the atoms whose reference metric is
    /// actually curvature-parameterised gives the flat layout exactly one raw
    /// `kappa` coordinate per estimand.
    pub kappa_atoms: Vec<usize>,
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
            log_lambda_block: Vec::new(),
            kappa: Vec::new(),
            kappa_atoms: Vec::new(),
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
            log_lambda_block: Vec::new(),
            kappa: Vec::new(),
            kappa_atoms: Vec::new(),
        }
    }

    /// Attach sectional curvatures as `(atom_index, kappa)` pairs. Empty
    /// restores the curvature-free layout. Atom indices must be strictly
    /// increasing so the mapping is canonical and flat-index lookup is exact.
    #[must_use]
    pub(crate) fn with_curvature(mut self, curvature: Vec<(usize, f64)>) -> Self {
        self.kappa_atoms = curvature.iter().map(|(atom, _)| *atom).collect();
        self.kappa = curvature.into_iter().map(|(_, value)| value).collect();
        self
    }

    /// Append the curvature state of a newly appended atom. Structural growth
    /// always adds atom `K-1`, so this preserves the canonical increasing map.
    pub(crate) fn append_curvature_atom(
        &mut self,
        atom: usize,
        kappa: Option<f64>,
    ) -> Result<(), String> {
        let Some(kappa) = kappa else {
            return Ok(());
        };
        if atom >= self.k_atoms() {
            return Err(format!(
                "cannot append curvature for atom {atom}: rho has K={}",
                self.k_atoms()
            ));
        }
        if self
            .kappa_atoms
            .last()
            .is_some_and(|&previous| previous >= atom)
        {
            return Err(format!(
                "cannot append curvature atom {atom} after {:?}",
                self.kappa_atoms.last()
            ));
        }
        if !kappa.is_finite() {
            return Err(format!("curvature for appended atom {atom} is not finite"));
        }
        self.kappa_atoms.push(atom);
        self.kappa.push(kappa);
        Ok(())
    }

    /// Apply an atom keep/remap permutation to the sparse curvature layout.
    /// Values follow their owning atoms; removed atoms lose their coordinate.
    pub(crate) fn remap_curvature_atoms(
        &mut self,
        old_to_new: &[Option<usize>],
    ) -> Result<(), String> {
        if self.kappa.len() != self.kappa_atoms.len() {
            return Err(format!(
                "cannot remap curvature: {} values but {} atom indices",
                self.kappa.len(),
                self.kappa_atoms.len()
            ));
        }
        let mut atoms = Vec::with_capacity(self.kappa_atoms.len());
        let mut values = Vec::with_capacity(self.kappa.len());
        for (&old_atom, &value) in self.kappa_atoms.iter().zip(self.kappa.iter()) {
            let mapped = old_to_new.get(old_atom).ok_or_else(|| {
                format!(
                    "cannot remap curvature atom {old_atom}: permutation has length {}",
                    old_to_new.len()
                )
            })?;
            if let Some(new_atom) = mapped {
                atoms.push(*new_atom);
                values.push(value);
            }
        }
        self.kappa_atoms = atoms;
        self.kappa = values;
        Ok(())
    }

    /// Bind the flat assignment-strength layout to the term's assignment
    /// family. The `K = 1` Softmax case, every hard-TopK case and an ordered
    /// Beta--Bernoulli prior whose concentration is effectively fixed are structural
    /// absences, not frozen coordinates.
    ///
    /// The layout reads the assignment, not only its mode: a per-fit concentration
    /// override pins a learnable mode, and that fit has no sparse coordinate
    /// ([`SaeAssignment::effective_alpha_is_learnable`], #2933 F45).
    #[must_use]
    pub fn for_assignment(mut self, assignment: &SaeAssignment) -> Self {
        self.assignment_strength_layout = AssignmentStrengthLayout::of(assignment);
        self
    }

    /// The flat outer-coordinate vector of this ρ for `assignment`: the only public
    /// flatten entry, and the one an outer problem's parameter count and seed come from.
    ///
    /// The constructors tag a ρ [`AssignmentStrengthLayout::PenaltyWeight`] until
    /// [`Self::for_assignment`] binds it, and the outer objective binds its own copy. A ρ whose
    /// layout disagrees with the assignment about the sparse coordinate would size the outer
    /// problem one coordinate away from the objective, so it is refused here, before any
    /// optimizer sees the vector (#2933 F45). The same holds for an ordered Beta--Bernoulli
    /// layout bound for the other concentration or for another family.
    pub fn to_flat(&self, assignment: &SaeAssignment) -> Result<Array1<f64>, String> {
        let bound = self.assignment_strength_layout;
        let expected = AssignmentStrengthLayout::of(assignment);
        let k_atoms = self.k_atoms();
        let ordered_contradiction = matches!(
            bound,
            AssignmentStrengthLayout::ConcentrationOffset
                | AssignmentStrengthLayout::FixedConcentration
        ) && bound != expected;
        if bound.has_outer_coordinate(k_atoms) != expected.has_outer_coordinate(k_atoms)
            || ordered_contradiction
        {
            let carries = |layout: AssignmentStrengthLayout| {
                if layout.has_outer_coordinate(k_atoms) {
                    "carries"
                } else {
                    "has no"
                }
            };
            return Err(format!(
                "SaeManifoldRho::to_flat: the rho layout {bound:?} {} sparse coordinate, but the \
                 {} assignment's layout {expected:?} {} one; bind the rho with \
                 `for_assignment(&assignment)` before flattening it (#2933 F45)",
                carries(bound),
                assignment.mode.family_label(),
                carries(expected)
            ));
        }
        Ok(self.flat_coordinates())
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
    pub(crate) fn smooth_flat_start(&self) -> usize {
        usize::from(self.sparse_flat_index().is_some())
    }

    /// Flat coordinate for atom `atom`'s smoothness strength.
    #[must_use]
    pub(crate) fn smooth_flat_index(&self, atom: usize) -> usize {
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

    /// Flat coordinates of the crosscoder block weights, consistent with
    /// [`Self::to_flat`]: after the ARD block and BEFORE the per-atom curvature
    /// tail. Empty for a plain SAE.
    #[must_use]
    pub(crate) fn block_flat_range(&self) -> std::ops::Range<usize> {
        let ard_len = self.log_ard.iter().map(|a| a.len()).sum::<usize>();
        let start = self.smooth_flat_start() + self.log_lambda_smooth.len() + ard_len;
        start..start + self.log_lambda_block.len()
    }

    /// Flat index of atom `k`'s sectional curvature, or `None` when this ρ
    /// carries no curvature coordinate (the historical case).
    ///
    /// Curvature is the LAST tail: after the ARD block and after the crosscoder
    /// block weights. Computing it as an offset from the end would be brittle
    /// the next time a tail is appended — the block gradient made exactly that
    /// mistake — so it is derived forwards from the same prefix arithmetic
    /// `ard_flat_index` uses.
    pub(crate) fn kappa_flat_index(&self, atom: usize) -> Option<usize> {
        let curvature_index = self.kappa_atoms.binary_search(&atom).ok()?;
        Some(self.block_flat_range().end + curvature_index)
    }

    /// Flat outer-coordinate index that atom `k`'s ARD axis `j` writes to,
    /// consistent with [`Self::to_flat`] / [`Self::from_flat`]: the unique
    /// coordinate `sparse_dim + K + Σ_{a<k} d_a + j`.
    #[must_use]
    pub fn ard_flat_index(&self, atom: usize, axis: usize) -> usize {
        let k = self.log_lambda_smooth.len();
        let base: usize = self.log_ard[..atom].iter().map(|a| a.len()).sum();
        self.smooth_flat_start() + k + base + axis
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
    /// coordinate is the log-concentration offset while α is effectively learnable, and
    /// absent otherwise (`SaeAssignment::ordered_beta_bernoulli_prior_parameters`);
    /// the seed leaves it unscaled. NONE of the ordered Beta--Bernoulli ρ coordinates therefore admit the
    /// Gaussian response-dispersion scaling; the seed stays at its absolute
    /// (already dimensionless) construction values, which keeps the smoothing/ARD
    /// penalties strong enough that the inner ordered Beta--Bernoulli solve cannot overfit at the seed
    /// and the EFS fixed point lands on the interior optimum instead of the
    /// zero-penalty collapse. The separable-gate modes are byte-for-byte
    /// unchanged.
    pub fn seed_scaled_by_dispersion_for_assignment(
        &self,
        dispersion: f64,
        assignment: &SaeAssignment,
    ) -> Result<Self, String> {
        let bound = self.clone().for_assignment(assignment);
        if matches!(assignment.mode, AssignmentMode::OrderedBetaBernoulli { .. }) {
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
        // Separable-gate modes (softmax entropy / ThresholdGate gated-L1): every
        // scale-coupled coordinate (the gate strength, each smoothness and each ARD
        // log-strength) takes the full shift `ln φ` at every K, so the seed's
        // effective stiffness `λ/φ` is the construction value (#3233).
        bound.seed_scaled_by_dispersion(dispersion)
    }

    fn seed_scaled_by_dispersion(&self, dispersion: f64) -> Result<Self, String> {
        if !(dispersion.is_finite() && dispersion > 0.0) {
            return Err(format!(
                "SaeManifoldRho::seed_scaled_by_dispersion: dispersion must be finite and \
                 positive; got {dispersion}"
            ));
        }
        let shift = dispersion.ln();
        let mut scaled = self.clone();
        if scaled.sparse_flat_index().is_some() {
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

    /// Physical assignment strength on the shared exact domain. This remains
    /// fallible because the public report fields may be edited after fitting;
    /// conversion always reads and validates current storage.
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
    /// The exact, unsaturated map revalidates current public storage.
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
    /// (#1556). The vector is returned only after every coordinate validates, so
    /// a caller never observes a partially converted table.
    #[must_use]
    pub(crate) fn lambda_smooth_vec(&self) -> Result<Vec<f64>, String> {
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
    pub fn ard_precisions(&self) -> Result<Vec<Array1<f64>>, String> {
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
        if self.kappa.len() != self.kappa_atoms.len() {
            return Err(format!(
                "curvature values length {} != curvature atom mapping length {}",
                self.kappa.len(),
                self.kappa_atoms.len()
            ));
        }
        let mut previous_atom = None;
        for (coordinate, (&atom, &value)) in
            self.kappa_atoms.iter().zip(self.kappa.iter()).enumerate()
        {
            if atom >= self.k_atoms() {
                return Err(format!(
                    "curvature coordinate {coordinate} names atom {atom}, outside K={}",
                    self.k_atoms()
                ));
            }
            if previous_atom.is_some_and(|previous| atom <= previous) {
                return Err(format!(
                    "curvature atom mapping must be strictly increasing; entry {coordinate} is {atom} after {previous_atom:?}"
                ));
            }
            if !value.is_finite() {
                return Err(format!(
                    "raw curvature at atom {atom} must be finite; got {value}"
                ));
            }
            previous_atom = Some(atom);
        }
        Ok(())
    }

    /// Generic objective-domain lower face in flat-rho layout. Log strengths
    /// share this exact endpoint; the owning SAE objective replaces raw `kappa`
    /// placeholders with scale-derived geometry rails.
    pub(crate) fn flat_domain_lower_bound(&self) -> Option<Array1<f64>> {
        let len = self.flat_coordinates().len();
        if len == 0 {
            return None;
        }
        Some(Array1::from_elem(len, LOG_STRENGTH_MIN))
    }

    /// Objective-domain upper face in flat-rho layout; see
    /// [`Self::flat_domain_lower_bound`].
    pub(crate) fn flat_domain_upper_bound(&self) -> Option<Array1<f64>> {
        let len = self.flat_coordinates().len();
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
    /// The `<ARD>` block concatenates each atom `k`'s per-axis `log_ard[k][j]`
    /// in atom order, axis `j` in `0..d_k`, so the length is
    /// `sparse_dim + K + Σ_k d_k` (plus the block and curvature tails).
    ///
    /// [`Self::from_flat`] is the exact inverse and reads the same layout from
    /// `self` (its `log_ard` shape).
    pub(crate) fn flat_coordinates(&self) -> Array1<f64> {
        let smooth_start = self.smooth_flat_start();
        let k = self.log_lambda_smooth.len();
        let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
        let block_len = self.log_lambda_block.len();
        let kappa_len = self.kappa.len();
        let mut out = Array1::<f64>::zeros(smooth_start + k + ard_len + block_len + kappa_len);
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
        // #2604 — per-atom curvature is APPENDED after the block tail,
        // by the same rule and for the same reason.
        for &v in &self.kappa {
            out[cursor] = v;
            cursor += 1;
        }
        out
    }

    /// Rebuild a ρ with this ρ's per-atom ARD dimensions from a flat
    /// outer-coordinate vector produced by [`Self::to_flat`].
    ///
    /// The per-atom dims are taken from `&self` (the ARD layout is a fixed
    /// property of the term shape; the engine only moves the values). The flat
    /// vector must have length `sparse_dim + K + Σ_k len(log_ard[k])` plus the
    /// block and curvature tails, where `K = len(log_lambda_smooth)` carries the
    /// per-atom smoothness coordinates (#1556).
    pub fn from_flat(&self, flat: ArrayView1<'_, f64>) -> Result<SaeManifoldRho, String> {
        let smooth_start = self.smooth_flat_start();
        let k = self.log_lambda_smooth.len();
        let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
        let block_len = self.log_lambda_block.len();
        let kappa_len = self.kappa.len();
        let expected = smooth_start + k + ard_len + block_len + kappa_len;
        if flat.len() != expected {
            return Err(format!(
                "SaeManifoldRho::from_flat: flat length {} != sparse_dim + K + \
                 Σ d_k + (L-1) = {expected}",
                flat.len()
            ));
        }
        let log_lambda_smooth: Vec<f64> = (0..k).map(|atom| flat[smooth_start + atom]).collect();
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
        let kappa: Vec<f64> = (0..kappa_len)
            .map(|b| flat[cursor + block_len + b])
            .collect();
        let rebuilt = SaeManifoldRho {
            log_lambda_sparse: self
                .sparse_flat_index()
                .map_or(self.log_lambda_sparse, |index| flat[index]),
            assignment_strength_layout: self.assignment_strength_layout,
            log_lambda_smooth,
            log_ard,
            log_lambda_block,
            kappa,
            kappa_atoms: self.kappa_atoms.clone(),
        };
        rebuilt.validate_log_strength_domain()?;
        Ok(rebuilt)
    }


}

#[cfg(test)]
mod curvature_coordinate_tests {
    use super::*;
    use ndarray::Array1;

    /// The curvature tail round-trips through the flat layout, and an EMPTY tail
    /// leaves that layout byte-identical.
    ///
    /// The second half is the load-bearing one: every dictionary without a
    /// constant-curvature atom must produce exactly the flat vector it produced
    /// before the coordinate existed, or adding it silently re-indexes every
    /// existing outer optimisation. That is the same contract
    /// `log_lambda_block` carries, checked the same way.
    #[test]
    fn curvature_tail_round_trips_and_is_absent_when_empty() {
        let base = SaeManifoldRho::new(-1.0, -2.0, vec![Array1::zeros(2), Array1::zeros(2)]);
        let without = base.flat_coordinates();

        let with_kappa = base.clone().with_curvature(vec![(0, 0.75), (1, -1.25)]);
        let flat = with_kappa.flat_coordinates();
        assert_eq!(
            flat.len(),
            without.len() + 2,
            "the curvature tail must extend the layout by exactly one entry per atom"
        );
        for i in 0..without.len() {
            assert_eq!(
                flat[i], without[i],
                "appending curvature must not disturb any earlier coordinate at index {i}"
            );
        }
        assert_eq!(flat[without.len()], 0.75);
        assert_eq!(flat[without.len() + 1], -1.25);

        let rebuilt = with_kappa.from_flat(flat.view()).unwrap();
        assert_eq!(rebuilt.kappa, vec![0.75, -1.25]);
        assert_eq!(rebuilt.kappa_atoms, vec![0, 1]);

        // Raw, not log: zero curvature is representable and round-trips, which is
        // the whole reason flat space is an interior point of the coordinate.
        let flat_zero = base
            .clone()
            .with_curvature(vec![(0, 0.0), (1, 0.0)])
            .flat_coordinates();
        let zero_rebuilt = base
            .clone()
            .with_curvature(vec![(0, 0.0), (1, 0.0)])
            .from_flat(flat_zero.view())
            .unwrap();
        assert_eq!(zero_rebuilt.kappa, vec![0.0, 0.0]);
    }
}
