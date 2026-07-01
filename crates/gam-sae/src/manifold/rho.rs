use super::*;

/// #1026 — how the per-atom ARD precisions are exposed to the OUTER REML
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
///   of K, so the outer optimizer searches `2 + max_d` hyperparameters. This is
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

/// REML-selected continuous hyperparameters for SAE-manifold.
#[derive(Debug, Clone)]
pub struct SaeManifoldRho {
    /// `log(lambda_sparse)` for softmax entropy or JumpReLU gated L1, or the
    /// learnable `log(alpha)` offset for IBP-MAP assignment.
    pub log_lambda_sparse: f64,
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
            log_lambda_smooth: vec![log_lambda_smooth; k],
            log_ard,
            ard_sharing: ArdSharing::PerAtom,
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
            log_lambda_smooth,
            log_ard,
            ard_sharing: ArdSharing::PerAtom,
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
            log_lambda_smooth: vec![log_lambda_smooth; k],
            log_ard,
            ard_sharing: ArdSharing::Shared,
        }
    }

    /// Largest per-atom ARD axis count `max_k d_k` (0 when ARD is disabled on
    /// every atom). This is the number of SHARED outer ARD coordinates in
    /// [`ArdSharing::Shared`] mode.
    #[must_use]
    pub fn max_ard_axes(&self) -> usize {
        self.log_ard.iter().map(|a| a.len()).max().unwrap_or(0)
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
    /// JumpReLU gated-L1), whose per-row gates are held at their seed weighting
    /// while the decoder/coordinates are refit, so `λ/φ` is exactly the effective
    /// stiffness.
    ///
    /// IBP-MAP is different in kind. Its per-row Bernoulli gates are FREE latent
    /// variables the inner joint solve co-optimizes with the coordinates and
    /// decoder. A response-dispersion-WEAKENED smoothness/ARD seed
    /// (`φ_seed ≪ 1` at any non-trivial noise scale) hands that extra gate +
    /// coordinate freedom enough slack to interpolate the noise: the inner solve
    /// overfits, the reconstruction dispersion `φ̂` collapses toward 0, and the
    /// Fellner–Schall multiplicative fixed point (`λ_new ∝ φ̂`) then spirals the
    /// smoothing/ARD penalties to zero — a degenerate outer basin the ρ-optimizer
    /// stalls in (#1744: ibp_map n=40 σ=0.18 stalled at EV 0.86). The IBP sparse
    /// coordinate is additionally a dimensionless log-alpha concentration offset,
    /// not a squared-output-unit penalty weight, so it was never dispersion-
    /// scalable either. NONE of the IBP-MAP ρ coordinates therefore admit the
    /// Gaussian response-dispersion scaling; the seed stays at its absolute
    /// (already dimensionless) construction values, which keeps the smoothing/ARD
    /// penalties strong enough that the inner IBP solve cannot overfit at the seed
    /// and the EFS fixed point lands on the interior optimum instead of the
    /// zero-penalty collapse. The separable-gate modes are byte-for-byte
    /// unchanged.
    pub fn seed_scaled_by_dispersion_for_assignment(
        &self,
        dispersion: f64,
        assignment_mode: AssignmentMode,
    ) -> Result<Self, String> {
        if matches!(assignment_mode, AssignmentMode::IBPMap { .. }) {
            // Validate the dispersion for parity with the scaled path (a
            // non-finite/​non-positive φ is still a caller error), then return the
            // unscaled seed: no IBP-MAP ρ coordinate is response-dispersion-scalable.
            if !(dispersion.is_finite() && dispersion > 0.0) {
                return Err(format!(
                    "SaeManifoldRho::seed_scaled_by_dispersion_for_assignment: dispersion must \
                     be finite and positive; got {dispersion}"
                ));
            }
            return Ok(self.clone());
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
        // Hessian indefinite — a non-PD seed whose Laplace evidence log-det is
        // undefined. Because the SAE fit runs a single seed (`max_seeds = 1`),
        // the EFS startup validation then rejects it with "no candidate seeds
        // passed outer startup validation" (the #1782 softmax / jumprelu failure),
        // exactly where ibp_map — which is never dispersion-weakened — survives.
        //
        // Fix: for K > 1 keep the seed decoder-smoothness / ARD from being
        // WEAKENED below their (dimensionless) construction strength — floor the
        // shift at 0 so noisy data (`φ > 1`) still STRENGTHENS smoothing (the
        // well-founded direction) while clean data can no longer collapse the
        // seed penalties into the non-PD basin. The sparse (gate) coordinate,
        // which does not enter the decoder Hessian, keeps its full dispersion
        // scaling. The EFS fixed point then descends each λ from this feasible,
        // PD seed to the same interior optimum.
        if self.log_lambda_smooth.len() <= 1 {
            return self.seed_scaled_by_dispersion_with_sparse_policy(dispersion, true);
        }
        if !(dispersion.is_finite() && dispersion > 0.0) {
            return Err(format!(
                "SaeManifoldRho::seed_scaled_by_dispersion_for_assignment: dispersion must \
                 be finite and positive; got {dispersion}"
            ));
        }
        let shift = dispersion.ln();
        let smooth_ard_shift = shift.max(0.0);
        let mut scaled = self.clone();
        scaled.log_lambda_sparse += shift;
        for value in &mut scaled.log_lambda_smooth {
            *value += smooth_ard_shift;
        }
        for atom in &mut scaled.log_ard {
            for value in atom.iter_mut() {
                *value += smooth_ard_shift;
            }
        }
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
        if scale_sparse {
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
        Ok(scaled)
    }

    pub fn lambda_sparse(&self) -> f64 {
        // Clamp the log-strength into the finite-normal band before
        // exponentiating: a raw `exp(log_lambda)` overflows to `inf` for
        // `log_lambda ≳ 709`, and `inf · 0.0` / `inf / inf` then injects NaN
        // into the penalty value/grad/Hessian and poisons the solve.
        Self::stable_exp_strength(self.log_lambda_sparse)
    }

    /// Number of atoms `K` carried by the per-atom smoothness vector.
    #[must_use]
    pub fn k_atoms(&self) -> usize {
        self.log_lambda_smooth.len()
    }

    /// Stable smoothness strength `exp(log_lambda_smooth[k])` for atom `k`
    /// (#1556). The exponent is clamped into the finite-normal band by
    /// [`Self::stable_exp_strength`] so the strength is always a finite,
    /// strictly-positive `f64`.
    #[must_use]
    pub fn lambda_smooth_for(&self, atom: usize) -> f64 {
        Self::stable_exp_strength(self.log_lambda_smooth[atom])
    }

    /// All `K` per-atom smoothness strengths `exp(log_lambda_smooth[k])`, atom
    /// order. Convenience for threading per-atom λ into the penalty assemblers
    /// (#1556).
    #[must_use]
    pub fn lambda_smooth_vec(&self) -> Vec<f64> {
        self.log_lambda_smooth
            .iter()
            .map(|&v| Self::stable_exp_strength(v))
            .collect()
    }

    /// Exponentiate a learnable log-strength with the exponent clamped into the
    /// finite-normal band, so the resulting strength is always a finite,
    /// strictly-positive `f64` (no overflow to `inf`, no underflow to `0.0`).
    pub(crate) fn stable_exp_strength(log_strength: f64) -> f64 {
        const MAX_LOG_STRENGTH: f64 = 700.0;
        const MIN_LOG_STRENGTH: f64 = -700.0;
        log_strength.clamp(MIN_LOG_STRENGTH, MAX_LOG_STRENGTH).exp()
    }

    /// Flatten ρ into the contiguous outer-coordinate vector the generic
    /// `OuterObjective` engine optimises over.
    ///
    /// Layout: `[log_lambda_sparse, <K smooth>, <ARD>]`, where `<K smooth>` is
    /// the per-atom `log_lambda_smooth[k]` in atom order (`k in 0..K`), so the
    /// smoothness block carries `K` outer coordinates, not 1 (#1556).
    ///
    /// * [`ArdSharing::PerAtom`] — the `<ARD>` block concatenates each atom
    ///   `k`'s per-axis `log_ard[k][j]` in atom order, axis `j` in `0..d_k`.
    ///   Empty per-atom blocks contribute no outer coordinates, so the length is
    ///   `1 + K + Σ_k d_k`.
    /// * [`ArdSharing::Shared`] — the `<ARD>` block is a constant `max_d =
    ///   max_k d_k` SHARED strengths, one per axis index `j`. Each shared value
    ///   is the mean of `log_ard[k][j]` over the atoms that own axis `j` (an
    ///   exact read-back when the table is already broadcast, which it always is
    ///   under this mode); the length is `1 + K + max_d` regardless of d.
    ///   (Smoothness stays per-atom in both modes; `ard_sharing` governs ARD
    ///   only.)
    ///
    /// [`Self::from_flat`] is the exact inverse and reads the same layout from
    /// `self` (its `log_ard` shape + `ard_sharing`).
    pub fn to_flat(&self) -> Array1<f64> {
        match self.ard_sharing {
            ArdSharing::PerAtom => {
                let k = self.log_lambda_smooth.len();
                let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
                let mut out = Array1::<f64>::zeros(1 + k + ard_len);
                out[0] = self.log_lambda_sparse;
                for (atom, &v) in self.log_lambda_smooth.iter().enumerate() {
                    out[1 + atom] = v;
                }
                let mut cursor = 1 + k;
                for axis in &self.log_ard {
                    for &v in axis.iter() {
                        out[cursor] = v;
                        cursor += 1;
                    }
                }
                out
            }
            ArdSharing::Shared => {
                let k = self.log_lambda_smooth.len();
                let max_d = self.max_ard_axes();
                let mut out = Array1::<f64>::zeros(1 + k + max_d);
                out[0] = self.log_lambda_sparse;
                for (atom, &v) in self.log_lambda_smooth.iter().enumerate() {
                    out[1 + atom] = v;
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
                    out[1 + k + j] = if count > 0 { acc / count as f64 } else { 0.0 };
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
    /// values). The flat vector must have length `1 + K + Σ_k len(log_ard[k])` in
    /// [`ArdSharing::PerAtom`] mode, or `1 + K + max_k d_k` in
    /// [`ArdSharing::Shared`] mode, where `K = len(log_lambda_smooth)` carries the
    /// per-atom smoothness coordinates (#1556) and the few shared per-axis ARD
    /// values are BROADCAST back to every atom that owns that axis, rebuilding the
    /// full per-atom table the inner solve consumes.
    pub fn from_flat(&self, flat: ArrayView1<'_, f64>) -> SaeManifoldRho {
        match self.ard_sharing {
            ArdSharing::PerAtom => {
                let k = self.log_lambda_smooth.len();
                let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
                assert_eq!(
                    flat.len(),
                    1 + k + ard_len,
                    "SaeManifoldRho::from_flat: flat length {} != 1 + K + Σ d_k = {}",
                    flat.len(),
                    1 + k + ard_len
                );
                let log_lambda_smooth: Vec<f64> = (0..k).map(|atom| flat[1 + atom]).collect();
                let mut log_ard = Vec::with_capacity(self.log_ard.len());
                let mut cursor = 1 + k;
                for axis in &self.log_ard {
                    let d = axis.len();
                    let mut block = Array1::<f64>::zeros(d);
                    for (j, slot) in block.iter_mut().enumerate() {
                        *slot = flat[cursor + j];
                    }
                    cursor += d;
                    log_ard.push(block);
                }
                SaeManifoldRho {
                    log_lambda_sparse: flat[0],
                    log_lambda_smooth,
                    log_ard,
                    ard_sharing: ArdSharing::PerAtom,
                }
            }
            ArdSharing::Shared => {
                let k = self.log_lambda_smooth.len();
                let max_d = self.max_ard_axes();
                assert_eq!(
                    flat.len(),
                    1 + k + max_d,
                    "SaeManifoldRho::from_flat: shared-ARD flat length {} != 1 + K + max_d = {}",
                    flat.len(),
                    1 + k + max_d
                );
                let log_lambda_smooth: Vec<f64> = (0..k).map(|atom| flat[1 + atom]).collect();
                // Broadcast the shared per-axis strengths into each atom's block,
                // preserving every atom's own `d_k` (a `d_k`-axis atom reads the
                // first `d_k` shared values). This rebuilds the full per-atom
                // table the construction / penalty assemblers read unchanged.
                let mut log_ard = Vec::with_capacity(self.log_ard.len());
                for axis in &self.log_ard {
                    let d = axis.len();
                    let mut block = Array1::<f64>::zeros(d);
                    for (j, slot) in block.iter_mut().enumerate() {
                        *slot = flat[1 + k + j];
                    }
                    log_ard.push(block);
                }
                SaeManifoldRho {
                    log_lambda_sparse: flat[0],
                    log_lambda_smooth,
                    log_ard,
                    ard_sharing: ArdSharing::Shared,
                }
            }
        }
    }
}
