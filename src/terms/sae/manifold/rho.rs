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
    /// `log(lambda_smooth)` shared by the per-atom decoder penalties.
    pub log_lambda_smooth: f64,
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
    #[must_use]
    pub fn new(log_lambda_sparse: f64, log_lambda_smooth: f64, log_ard: Vec<Array1<f64>>) -> Self {
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
        Self {
            log_lambda_sparse,
            log_lambda_smooth,
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

    /// Assignment-aware seed scaling. In learnable-alpha IBP mode the sparse
    /// coordinate is a dimensionless log-alpha offset, not a penalty strength, so
    /// response-dispersion scaling must skip it while still scaling smoothness and
    /// ARD precision seeds.
    pub fn seed_scaled_by_dispersion_for_assignment(
        &self,
        dispersion: f64,
        assignment_mode: AssignmentMode,
    ) -> Result<Self, String> {
        let scale_sparse = !matches!(
            assignment_mode,
            AssignmentMode::IBPMap {
                learnable_alpha: true,
                ..
            }
        );
        self.seed_scaled_by_dispersion_with_sparse_policy(dispersion, scale_sparse)
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
        scaled.log_lambda_smooth += shift;
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

    pub fn lambda_smooth(&self) -> f64 {
        Self::stable_exp_strength(self.log_lambda_smooth)
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
    /// Layout: `[log_lambda_sparse, log_lambda_smooth, <ARD>]`.
    ///
    /// * [`ArdSharing::PerAtom`] — the `<ARD>` block concatenates each atom
    ///   `k`'s per-axis `log_ard[k][j]` in atom order, axis `j` in `0..d_k`.
    ///   Empty per-atom blocks contribute no outer coordinates, so the length is
    ///   `2 + Σ_k d_k`.
    /// * [`ArdSharing::Shared`] — the `<ARD>` block is a constant `max_d =
    ///   max_k d_k` SHARED strengths, one per axis index `j`. Each shared value
    ///   is the mean of `log_ard[k][j]` over the atoms that own axis `j` (an
    ///   exact read-back when the table is already broadcast, which it always is
    ///   under this mode); the length is `2 + max_d` regardless of K.
    ///
    /// [`Self::from_flat`] is the exact inverse and reads the same layout from
    /// `self` (its `log_ard` shape + `ard_sharing`).
    pub fn to_flat(&self) -> Array1<f64> {
        match self.ard_sharing {
            ArdSharing::PerAtom => {
                let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
                let mut out = Array1::<f64>::zeros(2 + ard_len);
                out[0] = self.log_lambda_sparse;
                out[1] = self.log_lambda_smooth;
                let mut cursor = 2usize;
                for axis in &self.log_ard {
                    for &v in axis.iter() {
                        out[cursor] = v;
                        cursor += 1;
                    }
                }
                out
            }
            ArdSharing::Shared => {
                let max_d = self.max_ard_axes();
                let mut out = Array1::<f64>::zeros(2 + max_d);
                out[0] = self.log_lambda_sparse;
                out[1] = self.log_lambda_smooth;
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
                    out[2 + j] = if count > 0 { acc / count as f64 } else { 0.0 };
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
    /// values). The flat vector must have length `2 + Σ_k len(log_ard[k])` in
    /// [`ArdSharing::PerAtom`] mode, or `2 + max_k d_k` in [`ArdSharing::Shared`]
    /// mode (where the few shared per-axis values are BROADCAST back to every
    /// atom that owns that axis, rebuilding the full per-atom table the inner
    /// solve consumes).
    pub fn from_flat(&self, flat: ArrayView1<'_, f64>) -> SaeManifoldRho {
        match self.ard_sharing {
            ArdSharing::PerAtom => {
                let ard_len: usize = self.log_ard.iter().map(|a| a.len()).sum();
                assert_eq!(
                    flat.len(),
                    2 + ard_len,
                    "SaeManifoldRho::from_flat: flat length {} != 2 + Σ d_k = {}",
                    flat.len(),
                    2 + ard_len
                );
                let mut log_ard = Vec::with_capacity(self.log_ard.len());
                let mut cursor = 2usize;
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
                    log_lambda_smooth: flat[1],
                    log_ard,
                    ard_sharing: ArdSharing::PerAtom,
                }
            }
            ArdSharing::Shared => {
                let max_d = self.max_ard_axes();
                assert_eq!(
                    flat.len(),
                    2 + max_d,
                    "SaeManifoldRho::from_flat: shared-ARD flat length {} != 2 + max_d = {}",
                    flat.len(),
                    2 + max_d
                );
                // Broadcast the shared per-axis strengths into each atom's block,
                // preserving every atom's own `d_k` (a `d_k`-axis atom reads the
                // first `d_k` shared values). This rebuilds the full per-atom
                // table the construction / penalty assemblers read unchanged.
                let mut log_ard = Vec::with_capacity(self.log_ard.len());
                for axis in &self.log_ard {
                    let d = axis.len();
                    let mut block = Array1::<f64>::zeros(d);
                    for (j, slot) in block.iter_mut().enumerate() {
                        *slot = flat[2 + j];
                    }
                    log_ard.push(block);
                }
                SaeManifoldRho {
                    log_lambda_sparse: flat[0],
                    log_lambda_smooth: flat[1],
                    log_ard,
                    ard_sharing: ArdSharing::Shared,
                }
            }
        }
    }
}
