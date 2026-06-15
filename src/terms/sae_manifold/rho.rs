use super::*;

/// REML-selected continuous hyperparameters for SAE-manifold.
#[derive(Debug, Clone)]
pub struct SaeManifoldRho {
    /// `log(lambda_sparse)` for softmax entropy or JumpReLU gated L1, or the
    /// learnable `log(alpha)` offset for IBP-MAP assignment.
    pub log_lambda_sparse: f64,
    /// `log(lambda_smooth)` shared by the per-atom decoder penalties.
    pub log_lambda_smooth: f64,
    /// Per-atom, per-axis `log(alpha_kj)` ARD strengths. An empty per-atom
    /// block disables native coordinate ARD for that atom.
    pub log_ard: Vec<Array1<f64>>,
}

impl SaeManifoldRho {
    #[must_use]
    pub fn new(log_lambda_sparse: f64, log_lambda_smooth: f64, log_ard: Vec<Array1<f64>>) -> Self {
        Self {
            log_lambda_sparse,
            log_lambda_smooth,
            log_ard,
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
    /// Layout: `[log_lambda_sparse, log_lambda_smooth, <ARD>]`, where enabled
    /// ARD blocks concatenate each atom `k`'s per-axis `log_ard[k][j]` in atom
    /// order, axis `j` in `0..d_k`. Empty per-atom blocks contribute no outer
    /// coordinates. [`Self::from_flat`] is the exact inverse and reads this
    /// fixed per-atom layout from `self`.
    pub fn to_flat(&self) -> Array1<f64> {
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

    /// Rebuild a ρ with this ρ's per-atom ARD dimensions from a flat
    /// outer-coordinate vector produced by [`Self::to_flat`].
    ///
    /// The per-atom dims are taken from `&self` (the ARD layout is a fixed
    /// property of the term shape; the engine only moves the values). The
    /// flat vector must have length `2 + Σ_k len(log_ard[k])`.
    pub fn from_flat(&self, flat: ArrayView1<'_, f64>) -> SaeManifoldRho {
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
        }
    }
}
