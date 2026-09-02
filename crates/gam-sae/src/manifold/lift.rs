//! Lifted linear solvers for curved SAE atoms — *curvature is linear structure
//! one polynomial degree up*.
//!
//! # The pattern
//!
//! A curved atom fit by the dense per-row Newton over latent coordinates `t` is
//! nonconvex and basin-plagued. But every curved atom in this crate is, by
//! construction, a *linear* map applied to a fixed nonlinear feature map `Φ(t)`
//! (harmonic phasors, degree-2 monomials, …). Fitting the linear block is a
//! *convex* problem — the **lifted fit** — and recovering the underlying spike
//! parameters `{(a_j, t_j)}` from the fitted linear block is a closed-form
//! algebraic descent. [`crate::super_resolution`] is exactly this pattern for the
//! **circle**: a degree-`H` harmonic circle is a linear map on
//! `(cos t, …, cos Ht, sin t, …, sin Ht)`, and the matrix-pencil / Prony descent
//! un-superposes the fitted Fourier block into point masses. This module
//! generalises the descent to the two remaining curved topologies the term
//! carries ([`crate::manifold::SaeAtomBasisKind::Sphere`],
//! [`crate::manifold::SaeAtomBasisKind::Torus`]).
//!
//! * **Sphere — the Veronese lift.** A mixture of `m` points `v_1..v_m ∈ S^{d-1}`
//!   with amplitudes `a_j > 0` lifts to the PSD matrix `M = Σ_j a_j v_j v_jᵀ`
//!   (the degree-2 Veronese / symmetric-outer-product feature block). The descent
//!   is a symmetric eigendecomposition: [`recover_sphere_spikes`].
//! * **Torus — the Kronecker pencil.** A spike at `(θ, φ) ∈ T²` with per-axis
//!   harmonic degrees `(H₁, H₂)` lifts to the Kronecker product of the two
//!   harmonic phasor vectors; `m` spikes give a sum of `m` Kronecker-rank-1 terms
//!   sampled on the `H₁ × H₂` grid. The descent is 2-D harmonic retrieval by an
//!   enhanced matrix pencil with *auto-paired* axes: [`recover_torus_spikes`].
//!
//! Both descents are exact only in the noiseless limit; [`polish_spikes`] runs a
//! few damped Gauss–Newton steps on the *original* nonconvex objective
//! `‖z − Σ_j a_j Φ(t_j)‖²` given a caller-supplied basis evaluation, and reports
//! the final residual so a caller can gate acceptance.

// ============================================================================
// Shared order selection (mirrors `super_resolution`'s derived thresholds)
// ============================================================================

// ============================================================================
// Sphere — the Veronese lift
// ============================================================================

/// A single recovered point mass on the sphere `S^{d-1}`.
#[derive(Clone, Debug, PartialEq)]
pub struct SphereSpike {
    /// Canonical unit direction `v ∈ S^{d-1}` (length `d`). The lift `v vᵀ` is
    /// invariant under the antipodal flip `v ↦ −v`, so the reported vector is the
    /// canonical representative of the `{v, −v}` gauge orbit: its
    /// largest-magnitude component is non-negative (see [`canonicalize_direction`]).
    pub direction: Vec<f64>,
    /// Amplitude `a > 0` of the spike (the corresponding eigenvalue of the lift).
    pub amplitude: f64,
}

/// The full result of a Veronese-lift recovery.
#[derive(Clone, Debug)]
pub struct SphereRecovery {
    /// Recovered spikes, sorted by amplitude descending.
    pub spikes: Vec<SphereSpike>,
    /// Selected model order `m` (number of point masses), from the count of
    /// eigenvalues above the noise-derived floor.
    pub model_order: usize,
    /// Frobenius norm of `M̂ − Σ_j a_j v_j v_jᵀ` for the recovered model.
    pub residual: f64,
    /// Eigenvalues of the symmetrised lift, descending — the spectrum the order
    /// selection thresholded.
    pub eigenvalues: Vec<f64>,
}

// ============================================================================
// Torus — the Kronecker pencil (2-D harmonic retrieval, auto-paired)
// ============================================================================

/// A single recovered point mass on the torus `T² = S¹ × S¹`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorusSpike {
    /// Axis-1 position `θ ∈ [0, 1)` (fraction of a full turn); the phasor is
    /// `e^{2πi θ}`.
    pub theta: f64,
    /// Axis-2 position `φ ∈ [0, 1)`; the phasor is `e^{2πi φ}`.
    pub phi: f64,
    /// Amplitude `a > 0` (real part of the least-squares Kronecker-Vandermonde
    /// coefficient).
    pub amplitude: f64,
}

/// The full result of a Kronecker-pencil recovery.
#[derive(Clone, Debug)]
pub struct TorusRecovery {
    /// Recovered spikes, sorted by `(θ, φ)` ascending.
    pub spikes: Vec<TorusSpike>,
    /// Selected model order `m`.
    pub model_order: usize,
    /// Frobenius norm of `Ŷ − Σ_j a_j (z_j^{h₁} w_j^{h₂})` for the recovered
    /// model over the `H₁ × H₂` grid.
    pub residual: f64,
    /// Singular values of the enhanced (block-Hankel) matrix, descending.
    pub enhanced_singular_values: Vec<f64>,
}

// ============================================================================
// Polish — damped Gauss–Newton on the original nonconvex objective
// ============================================================================

/// Tuning for [`polish_spikes`].
#[derive(Clone, Debug)]
pub struct PolishOptions {
    /// Maximum outer Gauss–Newton iterations.
    pub max_iters: usize,
    /// Initial Levenberg–Marquardt damping `μ` (added as `μ‖δ‖²`). Small so the
    /// first step is nearly a pure Gauss–Newton step from a good pencil seed.
    pub initial_damping: f64,
    /// Stop when the residual norm improves by less than this between outer
    /// iterations (a local minimum has been reached to working precision).
    pub residual_tol: f64,
}

impl Default for PolishOptions {
    fn default() -> Self {
        Self {
            max_iters: 64,
            initial_damping: 1e-6,
            residual_tol: 1e-12,
        }
    }
}

/// Spike parameters in the original (un-lifted) coordinates: per-spike amplitude
/// and latent coordinate `t_j ∈ ℝ^d`.
#[derive(Clone, Debug)]
pub struct PolishState {
    /// Amplitudes `a_j`, one per spike.
    pub amplitudes: Vec<f64>,
    /// Latent coordinates `t_j`, `coords[j]` of length `d`.
    pub coords: Vec<Vec<f64>>,
}

/// Outcome of [`polish_spikes`].
#[derive(Clone, Debug)]
pub struct PolishResult {
    /// Polished parameters.
    pub state: PolishState,
    /// Final residual `‖z − Σ_j a_j Φ(t_j)‖₂`.
    pub residual: f64,
    /// Outer iterations actually taken.
    pub iterations: usize,
    /// `true` if the loop stopped on the residual-improvement tolerance or a
    /// damped step could no longer improve the residual (local optimum), `false`
    /// if it exhausted `max_iters`.
    pub converged: bool,
}

