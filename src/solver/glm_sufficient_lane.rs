//! Frozen-weight weighted-slab Gram tensor: the n-independence FIRST INSTANCE
//! for the design-moving GLM/PIRLS ψ-sweep (#1033 mechanism (c), first step).
//!
//! ## Where the Gaussian lane stops and this picks up
//!
//! The landed [`crate::solver::psi_gram_tensor::PsiGramTensor`] makes the
//! Gaussian-identity κ/ψ-sweep n-free for value AND `[ρ,ψ]`-gradient, because
//! the working weight `W` is CONSTANT there: `XᵀWX(ψ)` is a fixed bilinear in
//! the Chebyshev-in-ψ slabs of the design, so one O(n·k²) build pays the whole
//! ψ-sweep. For a non-Gaussian GLM that bilinear identity breaks: the working
//! weight is `W = W(ψ, β)` — it moves with the linear predictor
//! `η = X(ψ)β` every PIRLS iteration, so `XᵀW(ψ)X(ψ)` is NOT a fixed bilinear
//! in the design slabs and the Gaussian tensor cannot be reused as-is.
//! Mechanism (a) (see [`crate::solver::measure_jet_glm_sufficient`]) covers the
//! complementary case `dX/dψ == 0` (design fixed, only `W` moves); the open
//! frontier this module addresses is `dX/dψ ≠ 0` (design MOVING) for a GLM.
//!
//! ## The first correct, certified instance
//!
//! The outer REML loop sweeps ψ at a WARM β (the regime where the inner Newton
//! is near its previous solution). FREEZE the working weight `W` at the warm
//! working point. With `W` frozen, the weighted design
//!
//! ```text
//!   A(ψ) = diag(√w) · X(ψ)
//! ```
//!
//! is analytic in ψ on the trial window exactly as `X(ψ)` is (a fixed positive
//! diagonal scaling of analytic columns). Therefore
//!
//! ```text
//!   XᵀW X(ψ) = A(ψ)ᵀ A(ψ)
//! ```
//!
//! IS a fixed bilinear in the Chebyshev-in-ψ slabs of `A(ψ)` — which is exactly
//! the object [`PsiGramTensor`] already builds and certifies, here applied to
//! the WEIGHTED design `A(ψ)` with UNIT weights. So we get, n-free per ψ-trial
//! after one O(n·k²) build:
//!
//! ```text
//!   XᵀW X(ψ)       (the Fisher/expected-Hessian data-fit Gram at frozen W)
//!   XᵀW z̃(ψ)       z̃ = √w · (working response), folded into the same tensor
//!   ∂/∂ψ XᵀW X(ψ)  exact, same source of truth as the value
//! ```
//!
//! ### What this IS, exactly (honest scope — first step, not endgame)
//!
//! `XᵀW X(ψ)` at the FROZEN `W` is the EXACT expected-information (Fisher)
//! data-fit block for ONE Fisher-scoring step at any ψ in the window — i.e. the
//! Gram the first inner PIRLS iteration forms when it linearizes at the warm
//! working point. So a ψ-trial that holds this tensor can take its first inner
//! Newton/Fisher step n-free, and the warm-β outer-loop regime is precisely the
//! one where that first step is already close to convergence.
//!
//! It is NOT yet the fully-converged GLM REML objective at the new ψ: as the
//! inner loop iterates, `η = X(ψ)β` changes, so the true `W` drifts from the
//! frozen value. The certificate below bounds the frozen-`W` Gram against an
//! exact rebuild AT the frozen weights (so the tensor is bit-tight for what it
//! claims — the frozen-W Gram), and a separate caller-side guard (the
//! [`weight_drift_within`] tolerance) decides whether the warm `W` is still a
//! faithful stand-in for the trial's converged `W`; when the drift exceeds the
//! guard the caller falls back to the exact per-trial PIRLS rebuild. This is the
//! cleanest correct PIECE; the full mechanism-(c) endgame (a precision/SPDE
//! representation whose data-fit operator is ψ-free, paying the n-dependence
//! once for the CONVERGED objective) is documented in [`endgame_path`] below.
//!
//! ### Current wiring
//!
//! The per-trial n-pass for the GLM design-moving lane lives in
//! `SpatialJointContext::eval_full`
//! (`crate::terms::smooth::spatial_optimization`), the same seam the Gaussian
//! tensor plugs into: each θ=[ρ,ψ] trial re-realizes the n×k design at the new κ
//! and runs PIRLS. For eligible GLM families the installed path is:
//!
//!   1. After the outer loop's warm β is available, snapshot the working weight
//!      `w = w(η_warm)` and working response `z = z(η_warm)`.
//!   2. Build [`FrozenWeightGramTensor`] over the optimizer's ψ bounds with the
//!      existing exact realizer as `eval_design` (the SAME closure the Gaussian
//!      lane uses), pre-scaling by `√w` and folding `√w·z` as the response.
//!   3. In `eval_full`, when `tensor.contains(ψ)` AND `weight_drift_within(...)`
//!      holds for the trial's converged predictor, stage the first Fisher step's
//!      `XᵀWX` on the external evaluator so the inner PIRLS solve can consume it
//!      n-free; otherwise clear the slot and keep the exact rebuild.
//!
//! ### The GLM outer-GRADIENT n-free provider (#1033 (c), gradient channel)
//!
//! The value lane above (the first-Fisher-step `XᵀWX`) only retires the inner
//! P-IRLS first-iteration Gram build; the OUTER ψ-gradient still pays per-trial
//! n-contractions when it assembles the `j == 0` HyperCoord — the envelope
//! `a_j = −u·(X_τβ̂) + …`, the score `g_j = X_τᵀu − XᵀW(X_τβ̂) − S_τβ̂`. Both are
//! PURE first-order objects with NO third-derivative curvature term, so at the
//! converged β̂ they reduce EXACTLY (algebra identical to the Gaussian
//! `gaussian_psi_gram_deriv` lane) to the frozen-W tensor derivatives:
//!
//! ```text
//!   a_j = −(∂b/∂ψ·β̂ − ½β̂ᵀ(∂G/∂ψ)β̂) + ½β̂ᵀS_τβ̂
//!   g_j =  ∂b/∂ψ − (∂G/∂ψ)·β̂ − S_τβ̂
//! ```
//!
//! (using `u = W(z−η̂)` at the FROZEN `W`: `u·X_τβ̂ = ∂b/∂ψ·β̂ − ½β̂ᵀ∂G/∂ψ·β̂` and
//! `X_τᵀu − XᵀW X_τβ̂ = ∂b/∂ψ − ∂G/∂ψ·β̂`). [`FrozenWeightGramTensor::gradient_pair_if_sound`]
//! is the n-free provider for this channel: it returns `(∂G/∂ψ, ∂b/∂ψ)` only
//! when ψ is in the certified gradient sub-window AND the converged working
//! weight is within the TIGHT [`FrozenWeightGramTensor::GRADIENT_WEIGHT_DRIFT_RTOL`]
//! of the frozen snapshot — unlike the value lane (which RECONVERGES the true
//! `W`, so its value is exact at any drift), the gradient is read once at the
//! converged point with no safety net, so the gate must keep the frozen-W
//! derivative equal to the converged-W derivative to the outer-gradient bar.
//! The drift `B_j` (Hessian) is NOT served by this provider: GLM `B_j` carries
//! the irreducibly-n-dependent third-derivative curvature `Xᵀdiag(c⊙X_τβ̂)X`
//! that the frozen tensor does not hold, so the drift stays on the exact path.
//!
//! The integration lives in the spatial optimizer and external evaluator: they
//! snapshot the warm working weight, call `gradient_pair_if_sound`, and install
//! the resulting conditioned-frame pair onto the inner REML surface (a GLM
//! analogue of the Gaussian `install_gaussian_psi_gram_deriv` install). This
//! module is the n-free provider they consume, with both guards
//! ([`weight_drift_within`] for the value lane,
//! [`FrozenWeightGramTensor::GRADIENT_WEIGHT_DRIFT_RTOL`] for the gradient lane)
//! that keep the approximation honest.

use crate::solver::psi_gram_tensor::PsiGramTensor;
use ndarray::{Array1, Array2, ArrayView1};

/// Frozen-weight weighted-slab Gram tensor for the design-moving GLM ψ-sweep.
///
/// Wraps a [`PsiGramTensor`] built on the WEIGHTED design `A(ψ) = diag(√w)X(ψ)`
/// with the warm working weight `w` frozen, so every per-trial accessor is the
/// n-free Gaussian-shaped O(D²k²) assembly — but the assembled `gram_at(ψ)` is
/// the GLM Fisher data-fit Gram `XᵀW X(ψ)` at the frozen `W`. The frozen weight
/// vector is retained so the caller can certify, per trial, that the warm `W`
/// has not drifted past tolerance ([`weight_drift_within`]).
pub struct FrozenWeightGramTensor {
    inner: PsiGramTensor,
    /// The warm working weight `w` the tensor was frozen at. Retained for the
    /// per-trial drift guard, NOT consulted by the n-free accessors.
    frozen_w: Array1<f64>,
}

impl FrozenWeightGramTensor {
    /// Build and certify the frozen-`W` tensor over `ψ ∈ [psi_lo, psi_hi]`.
    ///
    /// `eval_design(ψ)` returns the EXACT unweighted n×k design `X(ψ)` (the same
    /// realizer the exact per-trial path uses). `frozen_w` is the warm working
    /// weight `w(η_warm)` (length n, finite, non-negative); `working_z` is the
    /// warm working response `z(η_warm)` (length n). The weighted design
    /// `A(ψ) = diag(√w)X(ψ)` and weighted response are formed once per node and
    /// handed to [`PsiGramTensor::build`] with UNIT weights, so the assembled
    /// `XᵀW X(ψ) = A(ψ)ᵀA(ψ)` is certified bit-tight against the exact rebuild
    /// AT the frozen weights. Returns `None` (caller keeps the exact path) when
    /// any input is degenerate / non-finite or no Chebyshev rung certifies.
    pub fn build(
        mut eval_design: impl FnMut(f64) -> Result<Array2<f64>, String>,
        frozen_w: ArrayView1<'_, f64>,
        working_z: ArrayView1<'_, f64>,
        psi_lo: f64,
        psi_hi: f64,
    ) -> Option<Self> {
        let n = frozen_w.len();
        if n == 0 || working_z.len() != n {
            return None;
        }
        if frozen_w.iter().any(|&w| !w.is_finite() || w < 0.0) {
            return None;
        }
        if working_z.iter().any(|&z| !z.is_finite()) {
            return None;
        }
        // Precompute √w once; the closure pre-scales each design row by it.
        let sqrt_w: Array1<f64> = frozen_w.mapv(f64::sqrt);
        // Weighted response z̃ = √w · z, folded as the PsiGramTensor "response"
        // with UNIT weights so that A(ψ)ᵀ z̃ = (√w X)ᵀ(√w z) = XᵀW z exactly,
        // and z̃ᵀz̃ = zᵀ W z (the Gaussian scalar sufficient statistic shape).
        let weighted_z: Array1<f64> =
            Array1::from_iter(working_z.iter().zip(sqrt_w.iter()).map(|(&z, &s)| z * s));
        let unit_weights: Array1<f64> = Array1::ones(n);

        let sqrt_w_closure = sqrt_w.clone();
        let weighted_eval = move |psi: f64| -> Result<Array2<f64>, String> {
            let mut design = eval_design(psi)?;
            if design.nrows() != sqrt_w_closure.len() {
                return Err(format!(
                    "frozen-W tensor: design has {} rows, expected {}",
                    design.nrows(),
                    sqrt_w_closure.len()
                ));
            }
            for (mut row, &s) in design.outer_iter_mut().zip(sqrt_w_closure.iter()) {
                row.mapv_inplace(|v| v * s);
            }
            Ok(design)
        };

        let inner = PsiGramTensor::build(
            weighted_eval,
            unit_weights.view(),
            weighted_z.view(),
            psi_lo,
            psi_hi,
        )
        .ok()?;
        Some(Self {
            inner,
            frozen_w: frozen_w.to_owned(),
        })
    }

    /// True when `ψ` lies inside the certified value window.
    pub fn contains(&self, psi: f64) -> bool {
        self.inner.contains(psi)
    }

    /// True when `ψ` lies inside the certified gradient sub-window (where the
    /// analytic `∂/∂ψ XᵀW X` is bit-tight against the exact frozen-W design
    /// derivative).
    pub fn contains_for_gradient(&self, psi: f64) -> bool {
        self.inner.contains_for_gradient(psi)
    }

    /// `XᵀW X(ψ)` at the frozen `W`, assembled n-free in O(D²k²) — the GLM
    /// Fisher data-fit Gram for the first Newton/Fisher step at `ψ`.
    pub fn gram_at(&self, psi: f64) -> Array2<f64> {
        self.inner.gram_at(psi)
    }

    /// `XᵀW z(ψ)` at the frozen `W`, n-free in O(Dk) — the GLM working RHS for
    /// the first Fisher step at `ψ`.
    pub fn rhs_at(&self, psi: f64) -> Array1<f64> {
        self.inner.rhs_at(psi)
    }

    /// Exact `∂/∂ψ (XᵀW X)` at the frozen `W`, n-free, from the SAME
    /// representation as the value — the cure for the objective↔gradient
    /// desync class on this channel.
    pub fn dgram_dpsi(&self, psi: f64) -> Array2<f64> {
        self.inner.dgram_dpsi(psi)
    }

    /// Exact `∂/∂ψ (XᵀW z)` at the frozen `W`, n-free.
    pub fn drhs_dpsi(&self, psi: f64) -> Array1<f64> {
        self.inner.drhs_dpsi(psi)
    }

    /// Exact `∂²/∂ψ² (XᵀW X)` at the frozen `W`, n-free in O(D²k²) — the
    /// frozen-W Fisher curvature block for the single-ψ Hessian channel. The
    /// GLM outer Hessian `B_j` carries an additional irreducibly-n-dependent
    /// third-derivative term (`Xᵀdiag(c⊙X_τβ̂)X`) that this tensor does not
    /// hold, so `B_j` stays on the exact slab (see the module header); this
    /// accessor exposes ONLY the n-free frozen-W Fisher second-ψ-derivative, in
    /// the same source-of-truth representation as the value and gradient.
    pub fn d2gram_dpsi2(&self, psi: f64) -> Array2<f64> {
        self.inner.d2gram_dpsi2(psi)
    }

    /// Exact `∂²/∂ψ² (XᵀW z)` at the frozen `W`, n-free.
    pub fn d2rhs_dpsi2(&self, psi: f64) -> Array1<f64> {
        self.inner.d2rhs_dpsi2(psi)
    }

    /// The frozen working weight the tensor was built at.
    pub fn frozen_weights(&self) -> ArrayView1<'_, f64> {
        self.frozen_w.view()
    }

    /// The n-free conditioned-frame ψ-gradient pair `(∂(XᵀWX)/∂ψ, ∂(XᵀWz)/∂ψ)`
    /// at the frozen `W`, returned ONLY when it is sound to serve the GLM outer
    /// gradient n-free for this trial — i.e. when BOTH
    ///
    ///   1. `ψ` lies inside the certified gradient sub-window
    ///      ([`Self::contains_for_gradient`]) where the analytic Chebyshev
    ///      derivative is bit-tight against the exact frozen-`W` design
    ///      derivative, AND
    ///   2. the trial's converged working weight `w_trial` is within the TIGHT
    ///      [`Self::GRADIENT_WEIGHT_DRIFT_RTOL`] of the frozen `W`, so the
    ///      frozen-`W` derivative IS the converged-`W` derivative to the
    ///      outer-gradient bar.
    ///
    /// Returns `None` otherwise — the caller then keeps the exact per-trial n×k
    /// `∂X/∂ψ` slab gradient. This is the GLM analogue of the Gaussian
    /// `gaussian_psi_gram_deriv` lane: the design-derivative half of the
    /// `j == 0` HyperCoord (`g_j`, the Fisher `B_j` drift) reduces EXACTLY to
    /// these two k-space objects at the converged point (Fisher curvature only —
    /// the same curvature the value channel's frozen Fisher step uses), so the
    /// per-trial `XᵀW(X_τβ̂)` and `X_τᵀu` n-passes are retired on this channel.
    pub fn gradient_pair_if_sound(
        &self,
        psi: f64,
        w_trial: ArrayView1<'_, f64>,
    ) -> Option<(Array2<f64>, Array1<f64>)> {
        if !self.contains_for_gradient(psi) {
            return None;
        }
        if !self.weight_drift_within(w_trial, Self::GRADIENT_WEIGHT_DRIFT_RTOL) {
            return None;
        }
        Some((self.dgram_dpsi(psi), self.drhs_dpsi(psi)))
    }

    /// Relative weight-drift tolerance for the GRADIENT channel.
    ///
    /// The value channel ([`Self::weight_drift_within`] at the caller's loose
    /// `FROZEN_GLM_WEIGHT_DRIFT_RTOL ≈ 1e-3`) can afford a loose gate because the
    /// inner P-IRLS RECONVERGES the true (moving) `W` after the frozen first
    /// Fisher step — so the trial's VALUE is exact regardless of drift; only the
    /// first Gram BUILD is skipped. The outer GRADIENT has no such safety net:
    /// it is read once at the converged point, so serving its design-derivative
    /// `(∂(XᵀWX)/∂ψ, ∂(XᵀWz)/∂ψ)` from the FROZEN `W` instead of the converged
    /// `W` injects a relative error of order `‖W_conv − W_frozen‖ / ‖W‖`
    /// directly into `∂G/∂ψ` (since `∂(XᵀWX)/∂ψ = X_τᵀW X + XᵀW X_τ` is LINEAR in
    /// `W`). The downstream outer REML gradient contracts `∂G/∂ψ` through `H⁻¹`
    /// and `β̂`, so to stay below the `1e-7` outer-gradient bar with margin (the
    /// same discipline as [`crate::solver::psi_gram_tensor::PSI_GRAM_SPOT_RTOL`]
    /// for the Gaussian lane) the gradient channel must only fire when
    /// the converged `W` is within a TIGHT relative drift of the frozen `W`.
    pub const GRADIENT_WEIGHT_DRIFT_RTOL: f64 = 1.0e-9;

    /// Per-trial honesty guard: true when the trial's converged working weight
    /// `w_trial` (formed from the new ψ's converged predictor) is within
    /// relative tolerance `rtol` of the frozen `w` the tensor was built at, so
    /// the frozen-`W` Gram is a faithful stand-in for the converged Gram.
    ///
    /// When this returns `false` the caller MUST fall back to the exact
    /// per-trial PIRLS rebuild — the frozen-`W` approximation is no longer
    /// trustworthy. This is the seam that keeps the first-instance lane honest:
    /// the tensor is certified bit-tight for the frozen-`W` Gram, and this guard
    /// certifies that the frozen `W` still represents the trial.
    pub fn weight_drift_within(&self, w_trial: ArrayView1<'_, f64>, rtol: f64) -> bool {
        if w_trial.len() != self.frozen_w.len() || !(rtol.is_finite() && rtol > 0.0) {
            return false;
        }
        // Relative max-norm drift ‖w_trial − w‖_∞ / (‖w‖_∞ + tiny).
        let w_scale = self
            .frozen_w
            .iter()
            .fold(0.0_f64, |acc, &w| acc.max(w.abs()))
            .max(1e-300);
        for (&wt, &w0) in w_trial.iter().zip(self.frozen_w.iter()) {
            if !wt.is_finite() {
                return false;
            }
            if (wt - w0).abs() > rtol * w_scale {
                return false;
            }
        }
        true
    }

    /// #1033 GLM endgame (value channel, n-free): the coefficient vector of ONE
    /// frozen-`W` Fisher-scoring step at `psi`, solving the penalized normal
    /// equations
    ///
    /// ```text
    ///   (XᵀW X(ψ) + S) β = XᵀW z̃(ψ)
    /// ```
    ///
    /// entirely in k-space — `XᵀW X(ψ)` from [`Self::gram_at`] and `XᵀW z̃(ψ)`
    /// from [`Self::rhs_at`], both assembled n-free from the certified tensor —
    /// so NO design row is touched. `penalty` is the (already ψ-keyed, k×k) total
    /// penalty `S` the caller supplies in the SAME coefficient basis the tensor
    /// was built in (the conditioned `x_fit` frame); it must be symmetric PSD so
    /// `XᵀW X(ψ) + S` is SPD for the Cholesky solve.
    ///
    /// This is the LOAD-BEARING math for serving a GLM κ-trial's inner solve
    /// without re-realizing the design: in the warm-β outer-loop regime the first
    /// Fisher step is already at (or near) the trial's converged β, so — gated by
    /// [`Self::weight_drift_within`] — this single k-space step IS the trial's
    /// inner solve. It is a PURE function (no `&mut`, no I/O): the wiring that
    /// routes the GLM lane through it (the design-revision skip) is a separate,
    /// independently-verified increment. The frozen-`W` Gram is bit-tight against
    /// the streamed weighted rebuild (see `frozen_w_gram_matches_exact_weighted_
    /// rebuild`), so β here equals the streamed first-Fisher-step β to the solve's
    /// conditioning.
    ///
    /// Returns `None` when `psi` is off-window, the penalty shape mismatches
    /// `k×k`, or the Cholesky factorization fails (non-SPD assembled system) —
    /// the caller then falls back to the exact per-trial PIRLS rebuild.
    pub fn frozen_w_single_step_beta(
        &self,
        psi: f64,
        penalty: ndarray::ArrayView2<'_, f64>,
    ) -> Option<Array1<f64>> {
        if !self.contains(psi) {
            return None;
        }
        let mut system = self.gram_at(psi);
        let k = system.nrows();
        if penalty.dim() != (k, k) {
            return None;
        }
        system += &penalty;
        if system.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let rhs = self.rhs_at(psi);
        if rhs.len() != k || rhs.iter().any(|v| !v.is_finite()) {
            return None;
        }
        use crate::faer_ndarray::FaerCholesky;
        let factor = system.cholesky(faer::Side::Lower).ok()?;
        let beta = factor.solvevec(&rhs);
        if beta.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(beta)
    }
}

/// The full mechanism-(c) endgame, documented for the next builder.
///
/// This module lands the FIRST correct piece: the frozen-`W` weighted-slab
/// tensor that serves the warm-β ψ-sweep's first Fisher step n-free. The
/// remaining path to the GLM n-independence ENDGAME, in increasing scope:
///
/// 1. **Per-PIRLS-iteration reuse (low-rank W correction).** Across inner
///    iterations only `diag(W)` changes (the design `X(ψ)` is frozen WITHIN a
///    ψ-trial). When the working-weight change between iterations is low-rank or
///    well-approximated by a rank-`q` diagonal correction `ΔW`, the data-fit
///    Gram updates as `XᵀWX ← XᵀWX + Xᵀ ΔW X`, where `Xᵀ ΔW X` is a rank-`q`
///    contraction — O(n·q·k) instead of O(n·k²) per iteration. The honest hook
///    is to express `ΔW = √(Δw)·√(Δw)ᵀ`-shaped diagonal updates against the
///    frozen-W slabs already stored here. (Distinct from this module's CROSS-ψ
///    reuse; this is WITHIN-ψ across IRLS steps.)
///
/// 2. **Joint (ψ, working-weight) tensor.** Tensor the slabs in BOTH ψ and a
///    scalar weight-summary coordinate so the converged `W(ψ)` is captured
///    analytically, retiring the frozen-`W` approximation. Requires certifying
///    a 2-D Chebyshev box and a contraction that respects the per-row coupling
///    `w_i = w_i(x_i(ψ)ᵀβ)` — a genuine research item.
///
/// 3. **Precision/SPDE representation (the real endgame).** A representation in
///    which the data-fit operator is ψ-FREE and only a k×k prior precision moves
///    with ψ (the Gaussian 1-D scan #1030 is the θ-free-operator instance). For
///    a GLM this is per-family and is the separate large build the issue calls
///    out — the converged objective becomes n-free, not just the first step.
///
/// This is a documentation-only marker so the path is discoverable from the
/// code that lands step one.
pub mod endgame_path {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Matérn-shaped synthetic design g(e^{u+ψ}), g(s)=(1+s)e^{−s}, plus a
    /// ψ-free power column — the exact structural mix of the radial designs,
    /// mirroring the PsiGramTensor oracle so we test the WEIGHTED composition.
    fn synth_design(psi: f64, n: usize, k: usize) -> Result<Array2<f64>, String> {
        let mut x = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                let r = 0.05 + (i as f64 + 1.0) * (j as f64 + 1.0) / (n as f64 * k as f64) * 3.0;
                if j == k - 1 {
                    x[[i, j]] = r * r * r;
                } else {
                    let s = r * psi.exp();
                    x[[i, j]] = (1.0 + s) * (-s).exp();
                }
            }
        }
        Ok(x)
    }

    /// A non-trivial positive working weight, varying per row like a real GLM
    /// Fisher weight (e.g. Bernoulli μ(1−μ)). Frozen across the ψ-sweep here.
    fn frozen_weights(n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |i| {
            let p = 0.1 + 0.8 * ((i as f64 + 0.5) / n as f64);
            p * (1.0 - p)
        })
    }

    fn working_z(n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |i| ((i as f64 * 0.37).sin()) + 0.5)
    }

    /// Exact XᵀW X at the frozen weights, rebuilt from rows — the reference.
    fn exact_weighted_gram(psi: f64, n: usize, k: usize, w: &Array1<f64>) -> Array2<f64> {
        let design = synth_design(psi, n, k).unwrap();
        let mut wd = design.clone();
        for (mut row, &wi) in wd.outer_iter_mut().zip(w.iter()) {
            row.mapv_inplace(|v| v * wi);
        }
        design.t().dot(&wd)
    }

    fn exact_weighted_xty(
        psi: f64,
        n: usize,
        k: usize,
        w: &Array1<f64>,
        z: &Array1<f64>,
    ) -> Array1<f64> {
        let design = synth_design(psi, n, k).unwrap();
        let mut out = Array1::<f64>::zeros(k);
        for i in 0..n {
            for j in 0..k {
                out[j] += design[[i, j]] * w[i] * z[i];
            }
        }
        out
    }

    #[test]
    fn frozen_w_gram_matches_exact_weighted_rebuild() {
        let (n, k) = (200usize, 5usize);
        let (psi_lo, psi_hi) = (-0.6, 0.6);
        let w = frozen_weights(n);
        let z = working_z(n);
        let tensor = FrozenWeightGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("frozen-W tensor must certify on the analytic Matérn-shaped design");

        // Off-node interior ψ values (never the build nodes).
        for &frac in &[0.137_f64, 0.382, 0.618, 0.851] {
            let psi = psi_lo + frac * (psi_hi - psi_lo);
            assert!(tensor.contains(psi));
            let assembled = tensor.gram_at(psi);
            let exact = exact_weighted_gram(psi, n, k, &w);
            let scale = exact
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()))
                .max(1e-300);
            for (a, b) in assembled.iter().zip(exact.iter()) {
                assert!(
                    (a - b).abs() <= 1e-9 * scale,
                    "XᵀWX(ψ={psi}) tensor vs exact frozen-W rebuild off by {}",
                    (a - b).abs()
                );
            }

            let rhs = tensor.rhs_at(psi);
            let exact_rhs = exact_weighted_xty(psi, n, k, &w, &z);
            let rscale = exact_rhs
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()))
                .max(1e-300);
            for (a, b) in rhs.iter().zip(exact_rhs.iter()) {
                assert!(
                    (a - b).abs() <= 1e-9 * rscale,
                    "XᵀWz(ψ={psi}) tensor vs exact off by {}",
                    (a - b).abs()
                );
            }
        }
    }

    /// #1033 GLM endgame value channel: the n-free frozen-`W` single Fisher step
    /// `(XᵀWX(ψ)+S)β = XᵀWz̃(ψ)` solved entirely from the tensor's k-space
    /// accessors must equal the SAME step solved from the streamed-exact
    /// frozen-`W` Gram+RHS (rebuilt from rows) — to the solve's conditioning.
    /// This pins the load-bearing math the GLM design-revision skip will route
    /// through, independent of any live-fit wiring.
    #[test]
    fn frozen_w_single_step_beta_matches_streamed_first_fisher_step() {
        let (n, k) = (200usize, 5usize);
        let (psi_lo, psi_hi) = (-0.6, 0.6);
        let w = frozen_weights(n);
        let z = working_z(n);
        let tensor = FrozenWeightGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("frozen-W tensor must certify on the analytic Matérn-shaped design");

        // A symmetric-PD penalty in the same k-space basis (diagonal ridge plus a
        // mild off-diagonal bend), chosen so XᵀWX(ψ)+S is comfortably SPD.
        let mut penalty = Array2::<f64>::zeros((k, k));
        for a in 0..k {
            penalty[[a, a]] = 0.5 + 0.1 * a as f64;
            if a + 1 < k {
                penalty[[a, a + 1]] = 0.05;
                penalty[[a + 1, a]] = 0.05;
            }
        }

        use crate::faer_ndarray::FaerCholesky;
        for &frac in &[0.137_f64, 0.382, 0.618, 0.851] {
            let psi = psi_lo + frac * (psi_hi - psi_lo);

            // n-free single step from the tensor accessors.
            let beta_nfree = tensor
                .frozen_w_single_step_beta(psi, penalty.view())
                .expect("frozen-W single step must solve (SPD system in-window)");

            // Streamed reference: assemble (XᵀWX+S) and XᵀWz̃ from rows, solve.
            let mut sys_ref = exact_weighted_gram(psi, n, k, &w);
            sys_ref += &penalty;
            let rhs_ref = exact_weighted_xty(psi, n, k, &w, &z);
            let beta_ref = sys_ref
                .cholesky(faer::Side::Lower)
                .expect("reference system SPD")
                .solvevec(&rhs_ref);

            let scale = beta_ref
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()))
                .max(1e-300);
            for (a, b) in beta_nfree.iter().zip(beta_ref.iter()) {
                assert!(
                    (a - b).abs() <= 1e-9 * scale,
                    "frozen-W single-step β(ψ={psi}) n-free vs streamed off by {}",
                    (a - b).abs()
                );
            }
        }

        // Shape-mismatched penalty and off-window ψ both refuse (sound fallback).
        let bad_penalty = Array2::<f64>::zeros((k + 1, k + 1));
        assert!(
            tensor
                .frozen_w_single_step_beta(0.0, bad_penalty.view())
                .is_none(),
            "a k+1×k+1 penalty must be refused"
        );
        assert!(
            tensor
                .frozen_w_single_step_beta(psi_hi + 1.0, penalty.view())
                .is_none(),
            "an off-window ψ must be refused"
        );
    }

    #[test]
    fn frozen_w_dgram_matches_finite_difference() {
        let (n, k) = (160usize, 4usize);
        let (psi_lo, psi_hi) = (-0.5, 0.5);
        let w = frozen_weights(n);
        let z = working_z(n);
        let tensor = FrozenWeightGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("frozen-W tensor must certify");

        // Only test where the gradient sub-window certifies.
        let psi = 0.0;
        if !tensor.contains_for_gradient(psi) {
            // The interior must certify for a smooth analytic design.
            panic!("gradient sub-window must certify at the window center");
        }
        let analytic = tensor.dgram_dpsi(psi);
        // 4th-order central FD of the exact frozen-W Gram.
        let h = 1e-4;
        let g = |p: f64| exact_weighted_gram(p, n, k, &w);
        let fd = (g(psi - 2.0 * h) - 8.0 * &g(psi - h) + 8.0 * &g(psi + h) - g(psi + 2.0 * h))
            / (12.0 * h);
        let scale = fd.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-300);
        for (a, b) in analytic.iter().zip(fd.iter()) {
            assert!(
                (a - b).abs() <= 1e-7 * scale,
                "∂(XᵀWX)/∂ψ analytic vs FD off by {}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn weight_drift_guard_accepts_warm_and_rejects_far() {
        let (n, k) = (120usize, 4usize);
        let w = frozen_weights(n);
        let z = working_z(n);
        let tensor = FrozenWeightGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            -0.4,
            0.4,
        )
        .expect("certify");

        // Identical weights: zero drift, accepted at any positive tolerance.
        assert!(tensor.weight_drift_within(w.view(), 1e-6));

        // A tiny perturbation within tolerance is accepted.
        let w_near: Array1<f64> = w.mapv(|v| v * (1.0 + 1e-4));
        assert!(tensor.weight_drift_within(w_near.view(), 1e-3));

        // A large drift is rejected → caller falls back to exact rebuild.
        let w_far: Array1<f64> = w.mapv(|v| v * 2.0);
        assert!(!tensor.weight_drift_within(w_far.view(), 1e-2));

        // A non-finite trial weight is rejected.
        let mut w_bad = w.clone();
        w_bad[0] = f64::NAN;
        assert!(!tensor.weight_drift_within(w_bad.view(), 1e-1));

        // Mismatched length is rejected.
        let w_short = Array1::<f64>::ones(n - 1);
        assert!(!tensor.weight_drift_within(w_short.view(), 1e-1));
    }

    #[test]
    fn gradient_pair_sound_only_in_window_and_under_tight_drift() {
        let (n, k) = (160usize, 4usize);
        let (psi_lo, psi_hi) = (-0.5, 0.5);
        let w = frozen_weights(n);
        let z = working_z(n);
        let tensor = FrozenWeightGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("frozen-W tensor must certify");

        let psi = 0.0;
        assert!(
            tensor.contains_for_gradient(psi),
            "interior must certify for the gradient lane"
        );

        // Zero drift (converged W == frozen W) inside the window: sound, and the
        // returned pair must equal the raw n-free derivatives (one source of
        // truth — no separate approximation injected by the gate).
        let pair = tensor
            .gradient_pair_if_sound(psi, w.view())
            .expect("zero-drift in-window trial must serve the gradient n-free");
        let dg = tensor.dgram_dpsi(psi);
        let db = tensor.drhs_dpsi(psi);
        for (a, b) in pair.0.iter().zip(dg.iter()) {
            assert_eq!(a, b, "gradient_pair ∂G/∂ψ must be the raw derivative");
        }
        for (a, b) in pair.1.iter().zip(db.iter()) {
            assert_eq!(a, b, "gradient_pair ∂b/∂ψ must be the raw derivative");
        }

        // A drift just above the tight gradient gate (but well inside the loose
        // value gate) must REFUSE the gradient lane — the gradient has no
        // reconvergence safety net, so the frozen-W derivative would inject
        // ~1e-6 relative error past the outer-gradient bar.
        let w_value_grade: Array1<f64> = w.mapv(|v| v * (1.0 + 1e-6)); // value gate (1e-3) accepts this
        assert!(
            tensor.weight_drift_within(w_value_grade.view(), 1e-3),
            "value-grade gate must accept a 1e-6 drift"
        );
        assert!(
            tensor
                .gradient_pair_if_sound(psi, w_value_grade.view())
                .is_none(),
            "gradient lane must refuse a drift past GRADIENT_WEIGHT_DRIFT_RTOL"
        );

        // A drift below the tight gradient gate is accepted.
        let w_grad_grade: Array1<f64> = w.mapv(|v| v * (1.0 + 1e-11));
        assert!(
            tensor
                .gradient_pair_if_sound(psi, w_grad_grade.view())
                .is_some(),
            "gradient lane must accept a drift within GRADIENT_WEIGHT_DRIFT_RTOL"
        );

        // Outside the gradient sub-window the lane refuses even at zero drift.
        let psi_edge = psi_hi - 1e-6;
        if !tensor.contains_for_gradient(psi_edge) {
            assert!(
                tensor.gradient_pair_if_sound(psi_edge, w.view()).is_none(),
                "near-edge ψ outside the gradient sub-window must refuse the lane"
            );
        }
    }

    #[test]
    fn rejects_degenerate_and_nonfinite_inputs() {
        let (n, k) = (50usize, 3usize);
        let w = frozen_weights(n);
        let z = working_z(n);

        // Empty weights → None.
        assert!(
            FrozenWeightGramTensor::build(
                |psi| synth_design(psi, n, k),
                Array1::<f64>::zeros(0).view(),
                Array1::<f64>::zeros(0).view(),
                -0.3,
                0.3,
            )
            .is_none()
        );

        // Negative weight → None (not a valid Fisher weight).
        let mut w_neg = w.clone();
        w_neg[1] = -0.1;
        assert!(
            FrozenWeightGramTensor::build(
                |psi| synth_design(psi, n, k),
                w_neg.view(),
                z.view(),
                -0.3,
                0.3,
            )
            .is_none()
        );

        // Mismatched z length → None.
        assert!(
            FrozenWeightGramTensor::build(
                |psi| synth_design(psi, n, k),
                w.view(),
                Array1::<f64>::zeros(n - 1).view(),
                -0.3,
                0.3,
            )
            .is_none()
        );

        // Degenerate window → None.
        assert!(
            FrozenWeightGramTensor::build(
                |psi| synth_design(psi, n, k),
                w.view(),
                z.view(),
                0.3,
                0.3,
            )
            .is_none()
        );
    }
}
