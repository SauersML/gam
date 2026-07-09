//! Criterion-as-atoms for the SAE LAML objective (issue #931, SAE pilot).
//!
//! # The bug class this kills
//!
//! The single most recurring structural bug in this engine is *objective↔
//! gradient desync*: the criterion **value** `V(ρ)` and its analytic
//! **derivatives** are assembled by separate code paths that drift apart
//! (#752, #748, #808, #901). `src/solver/reml/penalty_logdet.rs` already
//! contains the cure applied to one term — a single factorization emits the
//! value and every derivative as one coherent object, so they cannot disagree.
//! This module generalizes that shape to the **whole** SAE criterion as the
//! pilot for the GAM-wide migration (#931 migration note below).
//!
//! # The decomposition
//!
//! The SAE Laplace/REML criterion the outer optimizer minimizes is, exactly as
//! `SaeManifoldTerm::reml_criterion_with_cache` assembles it,
//!
//! ```text
//!   V(ρ) = [ loss.total() + extra_penalty_energy ]   (data-fit + priors)
//!        + ½·log|H(θ̂(ρ),ρ)|                          (Laplace logdet)
//!        − occam(ρ)                                   (smoothing Occam)
//! ```
//!
//! and its exact total ρ-derivative (the #1006 theorem, all coordinates at
//! once) is
//!
//! ```text
//!   dV/dρ = explicit               (∂/∂ρ of the data-fit+priors value)
//!         + logdet_trace           (½·tr(B⁻¹ ∂B/∂ρ))
//!         + occam_deriv            (−∂occam/∂ρ)
//!         + implicit_correction    (−½·Γᵀ A⁻¹ ∂g/∂ρ, the envelope/IFT term
//!                                   that accounts for θ̂(ρ) moving — #1006's Γ).
//! ```
//!
//! #1418: the criterion's Laplace curvature term is `½log|B|`, where `B` is the
//! curvature the inner solve factors (Gauss-Newton data curvature, softmax
//! Fisher metric, `max(V'',0)` ARD majorizers), so `Γ = ½tr(B⁻¹ ∂B/∂ρ)` and
//! `logdet_trace` contract `B`. But the implicit step `θ̂_ρ = −A⁻¹ g_ρ` is
//! governed by the EXACT stationarity Jacobian `A = ∇²_θθ L` (with residual
//! curvature, exact softmax entropy Hessian, exact periodic ARD curvature), NOT
//! the surrogate `B`. The implicit correction therefore solves against `A`
//! (`SaeManifoldTerm::solve_exact_stationarity`, a B⁻¹-preconditioned Neumann
//! fixed point on `A = B + ΔC`), so the correction is not biased by `B⁻¹ − A⁻¹`.
//!
//! # The atoms
//!
//! Each [`SaeCriterionAtom`] variant owns one of those terms and emits its
//! `(value, ρ-gradient)` **together** from the one cache the criterion forms.
//! [`SaeCriterion::assemble`] is the only constructor: it runs the inner solve
//! once, takes the single undamped factor, and hands every atom the SAME
//! `loss`/`log_det`/`cache`/`components` so value and gradient are projections
//! of one factorization — the outer optimizer (see
//! [`SaeCriterion::value`] / [`SaeCriterion::gradient`]) can no longer call a
//! value path and a gradient path that don't share state. The implicit-state
//! correction is its own atom rather than a hidden adjustment, so the envelope
//! term that the desync class repeatedly drops (#736's residue) is a named,
//! audited channel.
//!
//! # Migration note (#931, GAM-wide is follow-up scope)
//!
//! This pilots the atom composition on the SAE objective, where the #1006
//! analytic gradient and the #934 runtime certificate already exist to gate it.
//! The GAM-wide port (`src/solver/reml/unified.rs`: `LogdetHAtom`,
//! `PenaltyQuadAtom`, `JeffreysAtom`, family-correction atoms) is the documented
//! follow-up: each term leaves the monolith one atom at a time, FD-verified in
//! isolation, deleting its old value+gradient code in the same commit — no
//! parallel layers. `penalty_logdet.rs` needs no change; it is the template
//! every atom follows, and the #934 certificate is the runtime enforcement that
//! makes any un-ported residue observable on every fit.
//!
//! The `JeffreysAtom` shape named above is REALIZED for the SAE decoder
//! anti-collapse channel by [`super::manifold::penalties::BarrierComponent`]: the
//! decoder Jeffreys prior `−½·log det F(B)` (the `√det F` prior), assembled per
//! co-firing routed-support block with its value, gradient `−tr(F⁻¹ ∂F/∂B)`, and
//! self-concordant curvature emitted from one factorization — the same
//! single-source-of-truth contract this module documents. It is NOT a separate
//! atom variant here (the SAE β-tier is assembled by the arrow–Schur system, not
//! the `SaeCriterionAtom` enum), so there is no duplicate to maintain; the GAM-wide
//! `unified.rs::JeffreysAtom` remains the follow-up for the family-information
//! Jeffreys term on the coefficient tier.

use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};

/// Per-row / per-atom energy floor shared by the residual-EM forward and its
/// VJP. Matches the `clamp_min(1e-12)` guards on the norms in the torch lane
/// (`ManifoldSAE.reconstruction_topk_gate`) so the Rust kernel and the deleted
/// Python tensor algebra agree bit-for-bit on degenerate near-zero rows/atoms.
const RESIDUAL_EM_ENERGY_FLOOR: f64 = 1e-12;

/// Residual-EM routing scores for the torch `softmax_topk` lane (issue #1282),
/// ported out of `ManifoldSAE.reconstruction_topk_gate` so the Python side only
/// marshals tensors. This is the load-bearing *criterion* math the gate scores
/// atoms with; the surrounding routing orchestration (Sinkhorn balance, cluster
/// / quadratic anchors, the assignment EMA, the top-k selection mask and the
/// straight-through blend) is value-only control flow that already lives in Rust
/// (`gam::geometry::sae_routing`) or is a torch primitive, so it stays on the
/// Python side.
///
/// # The criterion
///
/// For each row `x ∈ ℝ^d` and atom `f` with current decoded curve
/// `r = per_atom_recon[·, f, ·] ∈ ℝ^d` we solve the best scalar code against
/// that atom's curve and score the atom by the relative residual it leaves:
///
/// ```text
///   D          = max(‖r‖², ε)                        (atom energy, floored)
///   s          = (r·x) / D                            (least-squares coefficient)
///   c          = max(s, 0)  [nonneg]   or  s  [signed]
///   e          = c·r − x                              (residual vector)
///   R          = ‖e‖²                                 (reconstruction residual)
///   ρ          = max(‖x‖², ε)                         (row energy, floored)
///   q          = R / ρ                                (scale-free relative residual)
/// ```
///
/// The `nonneg` code is the `target_k == 1` convention (the deterministic-
/// annealing EM responsibilities `softmax(−q/τ)` are built from `q` and the
/// routed magnitude is the non-negative `c`); the `signed` code is the
/// `target_k > 1` convention (the signed least-squares coefficient both scores
/// and gates so the effective active count tracks `target_k`). One kernel with a
/// flag emits the matched `(c, q)` pair for whichever convention the caller uses.
///
/// Returns `(code (n, f), relative_residual (n, f))`.
#[must_use]
pub fn residual_em_score(
    x: ArrayView2<'_, f64>,
    per_atom_recon: ArrayView3<'_, f64>,
    nonneg: bool,
) -> (Array2<f64>, Array2<f64>) {
    let (n, f, d) = per_atom_recon.dim();
    assert_eq!(
        x.dim(),
        (n, d),
        "residual_em_score: x is {:?} but per_atom_recon is {:?}",
        x.dim(),
        (n, f, d)
    );
    let mut code = Array2::<f64>::zeros((n, f));
    let mut relative_residual = Array2::<f64>::zeros((n, f));
    for row in 0..n {
        let mut row_energy = 0.0;
        for j in 0..d {
            let xj = x[[row, j]];
            row_energy += xj * xj;
        }
        let row_scale = row_energy.max(RESIDUAL_EM_ENERGY_FLOOR);
        for atom in 0..f {
            let mut rr = 0.0;
            let mut rx = 0.0;
            for j in 0..d {
                let rj = per_atom_recon[[row, atom, j]];
                rr += rj * rj;
                rx += rj * x[[row, j]];
            }
            let denom = rr.max(RESIDUAL_EM_ENERGY_FLOOR);
            let s = rx / denom;
            let c = if nonneg { s.max(0.0) } else { s };
            let mut resid = 0.0;
            for j in 0..d {
                let e = c * per_atom_recon[[row, atom, j]] - x[[row, j]];
                resid += e * e;
            }
            code[[row, atom]] = c;
            relative_residual[[row, atom]] = resid / row_scale;
        }
    }
    (code, relative_residual)
}

/// Vector–Jacobian product (analytic backward) for [`residual_em_score`], w.r.t.
/// `per_atom_recon`. This is the torch-lane gradient channel: `code` and
/// `relative_residual` both carry reconstruction gradient into the decoder
/// (`code` is the routed magnitude and `q = R/ρ` feeds the soft EM
/// responsibilities), so the cutover keeps the tape continuous by pairing this
/// with the Rust forward inside a `torch.autograd.Function`. `x` is the
/// activation batch — a constant on the tape (it never requires grad), so no
/// `∂/∂x` channel is produced.
///
/// # Derivation
///
/// Fix one `(row, atom)` with curve `r`, coefficient `s = (r·x)/D`,
/// `D = max(‖r‖², ε)`, code `c`, residual vector `e = c·r − x`, `R = ‖e‖²`,
/// row scale `ρ = max(‖x‖², ε)` and relative residual `q = R/ρ`. Given upstream
/// `g_c = ∂L/∂c` and `g_q = ∂L/∂q`, the chain rule gives `∂L/∂r = g_c·∂c/∂r +
/// g_q·∂q/∂r`.
///
/// * Coefficient. With `D` in its differentiable branch (`‖r‖² ≥ ε`, so
///   `∂D/∂r = 2r`), `∂s/∂r = x/D − 2·s·r/D = (x − 2·s·r)/D`. On the floored
///   branch `∂D/∂r = 0`, so `∂s/∂r = x/D`. The `nonneg` clamp passes gradient
///   exactly where torch's `clamp_min(0)` does — where `s ≥ 0`:
///   `∂c/∂r = active·∂s/∂r`, `active = (s ≥ 0)` (nonneg) or `1` (signed).
/// * Residual. `R = ‖c·r − x‖²` with `c = c(r)`, so
///   `∂R/∂r = 2·(e·r)·∂c/∂r + 2·c·e`, and `∂q/∂r = (1/ρ)·∂R/∂r`.
///
/// Collecting the two channels,
///
/// ```text
///   ∂L/∂r = A · ∂s/∂r + (2·g_q·c/ρ) · e,
///   A     = active · ( g_c + 2·g_q·(e·r)/ρ ).
/// ```
///
/// When `active` is false the code is clamped to `0`, so `c = 0` kills the `e`
/// term too and the whole gradient vanishes — the flat interior of the clamp,
/// matching torch exactly.
///
/// Returns `grad_per_atom_recon (n, f, d)`.
#[must_use]
pub fn residual_em_score_vjp(
    x: ArrayView2<'_, f64>,
    per_atom_recon: ArrayView3<'_, f64>,
    nonneg: bool,
    g_code: ArrayView2<'_, f64>,
    g_relative_residual: ArrayView2<'_, f64>,
) -> Array3<f64> {
    let (n, f, d) = per_atom_recon.dim();
    assert_eq!(x.dim(), (n, d), "residual_em_score_vjp: x shape mismatch");
    assert_eq!(g_code.dim(), (n, f), "residual_em_score_vjp: g_code shape");
    assert_eq!(
        g_relative_residual.dim(),
        (n, f),
        "residual_em_score_vjp: g_relative_residual shape"
    );
    let mut grad = Array3::<f64>::zeros((n, f, d));
    for row in 0..n {
        let mut row_energy = 0.0;
        for j in 0..d {
            let xj = x[[row, j]];
            row_energy += xj * xj;
        }
        let row_scale = row_energy.max(RESIDUAL_EM_ENERGY_FLOOR);
        for atom in 0..f {
            let mut rr = 0.0;
            let mut rx = 0.0;
            for j in 0..d {
                let rj = per_atom_recon[[row, atom, j]];
                rr += rj * rj;
                rx += rj * x[[row, j]];
            }
            // `clamp_min` passes gradient through its argument iff it is on the
            // un-floored branch (`‖r‖² ≥ ε`); mirror that for `∂D/∂r`.
            let denom_active = rr >= RESIDUAL_EM_ENERGY_FLOOR;
            let denom = rr.max(RESIDUAL_EM_ENERGY_FLOOR);
            let s = rx / denom;
            let active = if nonneg { s >= 0.0 } else { true };
            let c = if nonneg { s.max(0.0) } else { s };
            let mut e_dot_r = 0.0;
            for j in 0..d {
                let e = c * per_atom_recon[[row, atom, j]] - x[[row, j]];
                e_dot_r += e * per_atom_recon[[row, atom, j]];
            }
            let gc = g_code[[row, atom]];
            let gq = g_relative_residual[[row, atom]];
            let a = if active {
                gc + 2.0 * gq * e_dot_r / row_scale
            } else {
                0.0
            };
            let coeff_e = 2.0 * gq * c / row_scale;
            for j in 0..d {
                let rj = per_atom_recon[[row, atom, j]];
                let ds_drj = if denom_active {
                    (x[[row, j]] - 2.0 * s * rj) / denom
                } else {
                    x[[row, j]] / denom
                };
                let e = c * rj - x[[row, j]];
                grad[[row, atom, j]] = a * ds_drj + coeff_e * e;
            }
        }
    }
    grad
}

/// One additive term of the SAE LAML criterion, carrying its scalar value and
/// its ρ-gradient contribution from the **same** emission so the two cannot
/// drift. The variants partition `V(ρ)` and `dV/dρ` term for term.
#[derive(Debug, Clone)]
pub enum SaeCriterionAtom {
    /// `loss.total() + extra_penalty_energy`, the penalized deviance the inner
    /// Newton solve descends, with its explicit ρ-derivative (assignment-prior
    /// log-strength, smoothness penalty energy, ARD log-precision).
    DataFitPriors {
        /// `loss.total() + extra_penalty_energy`.
        value: f64,
        /// Explicit ∂/∂ρ of the data-fit + priors value.
        grad: Array1<f64>,
    },
    /// `½·log|H(θ̂,ρ)|`, the Laplace normaliser, with `½·tr(H⁻¹ ∂H/∂ρ)`. The
    /// logdet is defined on the criterion's H (the PSD-majorized assembly), and
    /// the trace differentiates THE SAME object on the same smooth branch — the
    /// #1006 landmine-1 single-source-of-truth contract.
    LaplaceLogdet {
        /// `½·log|H|`.
        value: f64,
        /// `½·tr(H⁻¹ ∂H/∂ρ)`.
        grad: Array1<f64>,
    },
    /// `−occam(ρ)`, the smoothing-penalty Occam term, with `−∂occam/∂ρ`.
    Occam {
        /// `−occam(ρ)`.
        value: f64,
        /// `−∂occam/∂ρ`.
        grad: Array1<f64>,
    },
    /// The implicit-state envelope correction `−½·Γᵀ H⁻¹ ∂g/∂ρ` (#1006's Γ):
    /// the part of `dV/dρ` arising because the inner optimum `θ̂(ρ)` moves with
    /// ρ. It contributes **no value** (the value terms are evaluated at the
    /// converged θ̂; the envelope theorem kills `∂L/∂θ·θ̂'`) — only a gradient
    /// channel. It is a named atom precisely because this is the channel the
    /// desync class keeps dropping; making it explicit makes its omission a
    /// visible missing atom, not a silent zero.
    ImplicitStationarityCorrection {
        /// `−½·Γᵀ H⁻¹ ∂g/∂ρ`.
        grad: Array1<f64>,
    },
}

impl SaeCriterionAtom {
    /// This atom's contribution to the criterion **value** `V(ρ)`. Pure-gradient
    /// atoms contribute `0.0`.
    #[must_use]
    pub fn value(&self) -> f64 {
        match self {
            Self::DataFitPriors { value, .. }
            | Self::LaplaceLogdet { value, .. }
            | Self::Occam { value, .. } => *value,
            Self::ImplicitStationarityCorrection { .. } => 0.0,
        }
    }

    /// This atom's contribution to the criterion **ρ-gradient** `dV/dρ`.
    #[must_use]
    pub fn grad(&self) -> &Array1<f64> {
        match self {
            Self::DataFitPriors { grad, .. }
            | Self::LaplaceLogdet { grad, .. }
            | Self::Occam { grad, .. }
            | Self::ImplicitStationarityCorrection { grad } => grad,
        }
    }

    /// A short stable label for the atom, used by the consistency audit and any
    /// desync diagnostic to name the offending term.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::DataFitPriors { .. } => "data_fit_priors",
            Self::LaplaceLogdet { .. } => "laplace_logdet",
            Self::Occam { .. } => "occam",
            Self::ImplicitStationarityCorrection { .. } => "implicit_stationarity_correction",
        }
    }
}

/// The SAE LAML criterion as a sum of atoms. The outer optimizer consumes only
/// `Σ atoms` through [`Self::value`] and [`Self::gradient`]; because every atom
/// was emitted from one cache by [`Self::assemble`], the value and gradient are
/// projections of a single factorization and cannot disagree by construction.
#[derive(Debug, Clone)]
pub struct SaeCriterion {
    atoms: Vec<SaeCriterionAtom>,
    n_rho: usize,
}

impl SaeCriterion {
    /// Compose the criterion from the four atoms. All four are built from the
    /// SAME `loss`/`log_det`/gradient-component emission so this is the only
    /// place the criterion's value-vs-gradient coherence is established.
    ///
    /// `data_fit_priors_value` is `loss.total() + extra_penalty_energy`;
    /// `log_det` is `log|H|`; `occam` is the smoothing Occam term (the criterion
    /// subtracts it). The gradient component arrays are the
    /// `SaeOuterRhoGradientComponents` channels.
    #[must_use]
    pub fn assemble(
        data_fit_priors_value: f64,
        log_det: f64,
        occam: f64,
        explicit: Array1<f64>,
        logdet_trace: Array1<f64>,
        occam_grad: Array1<f64>,
        implicit_correction: Array1<f64>,
    ) -> Self {
        let n_rho = explicit.len();
        let atoms = vec![
            SaeCriterionAtom::DataFitPriors {
                value: data_fit_priors_value,
                grad: explicit,
            },
            SaeCriterionAtom::LaplaceLogdet {
                value: 0.5 * log_det,
                grad: logdet_trace,
            },
            SaeCriterionAtom::Occam {
                value: -occam,
                grad: occam_grad,
            },
            SaeCriterionAtom::ImplicitStationarityCorrection {
                grad: implicit_correction,
            },
        ];
        Self { atoms, n_rho }
    }

    /// The criterion value `V(ρ) = Σ atoms.value()`.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.atoms.iter().map(SaeCriterionAtom::value).sum()
    }

    /// The criterion gradient `dV/dρ = Σ atoms.grad()`.
    #[must_use]
    pub fn gradient(&self) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.n_rho);
        for atom in &self.atoms {
            out += atom.grad();
        }
        out
    }

    /// The atoms, for per-term inspection / the desync diagnostic.
    #[must_use]
    pub fn atoms(&self) -> &[SaeCriterionAtom] {
        &self.atoms
    }

    /// Number of ρ coordinates.
    #[must_use]
    pub fn n_rho(&self) -> usize {
        self.n_rho
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample_criterion() -> SaeCriterion {
        SaeCriterion::assemble(
            3.0,                       // data-fit + priors value
            2.0,                       // log|H|
            0.5,                       // occam
            array![0.10, -0.20, 0.05], // explicit grad
            array![0.01, 0.02, -0.03], // logdet trace
            array![-0.04, 0.00, 0.06], // occam grad (−∂occam/∂ρ)
            array![0.07, -0.01, 0.00], // implicit correction
        )
    }

    /// The composite value is exactly the sum of the atom values, and matches
    /// the hand-assembled `loss + ½log|H| − occam`.
    #[test]
    fn value_is_atom_sum() {
        let crit = sample_criterion();
        let expected = 3.0 + 0.5 * 2.0 - 0.5;
        assert!((crit.value() - expected).abs() < 1e-12);
        // Atom-by-atom partition reproduces the total.
        let by_atom: f64 = crit.atoms().iter().map(SaeCriterionAtom::value).sum();
        assert!((by_atom - expected).abs() < 1e-12);
    }

    /// The composite gradient is exactly the sum of the four channels — the
    /// implicit-correction atom is included, so the gradient is the corrected
    /// #1006 total, never the uncorrected one.
    #[test]
    fn gradient_is_channel_sum_including_correction() {
        let crit = sample_criterion();
        let g = crit.gradient();
        let expected = array![
            0.10 + 0.01 - 0.04 + 0.07,
            -0.20 + 0.02 + 0.00 - 0.01,
            0.05 - 0.03 + 0.06 + 0.00
        ];
        for i in 0..3 {
            assert!(
                (g[i] - expected[i]).abs() < 1e-12,
                "coord {i}: {} vs {}",
                g[i],
                expected[i]
            );
        }
    }

    /// The implicit-stationarity atom carries no value (envelope theorem) but a
    /// real gradient — the structural property that makes a dropped envelope
    /// term a visible missing atom rather than a silent zero.
    #[test]
    fn implicit_correction_atom_is_gradient_only() {
        let atom = SaeCriterionAtom::ImplicitStationarityCorrection {
            grad: array![1.0, 2.0, 3.0],
        };
        assert_eq!(atom.value(), 0.0);
        assert_eq!(atom.grad().sum(), 6.0);
        assert_eq!(atom.label(), "implicit_stationarity_correction");
    }

    /// Residual-EM forward + analytic VJP on a hand-computable `n=f=1, d=2`
    /// case. `x = [2, 1]`, `r = [1, 0]`:
    ///   `D = 1`, `s = 2`, `c = 2` (both conventions since `s > 0`),
    ///   `e = c·r − x = [0, −1]`, `R = 1`, `ρ = 5`, `q = 0.2`.
    /// Code channel (`g_c = 1, g_q = 0`): `∂c/∂r = (x − 2s·r)/D = [−2, 1]`.
    /// Relative-residual channel (`g_c = 0, g_q = 1`):
    ///   `∂R/∂r = 2(e·r)∂c/∂r + 2c·e = [0, −4]`, so `∂q/∂r = [0, −0.8]`.
    #[test]
    fn residual_em_score_matches_hand_computation() {
        let x = array![[2.0, 1.0]];
        let recon =
            ndarray::Array3::from_shape_vec((1, 1, 2), vec![1.0, 0.0]).unwrap();

        for nonneg in [true, false] {
            let (code, relres) = residual_em_score(x.view(), recon.view(), nonneg);
            assert!((code[[0, 0]] - 2.0).abs() < 1e-12, "code ({nonneg})");
            assert!((relres[[0, 0]] - 0.2).abs() < 1e-12, "relres ({nonneg})");

            // Code channel only.
            let g_c = array![[1.0]];
            let g_q = array![[0.0]];
            let grad =
                residual_em_score_vjp(x.view(), recon.view(), nonneg, g_c.view(), g_q.view());
            assert!((grad[[0, 0, 0]] - (-2.0)).abs() < 1e-12, "dc/dr0 ({nonneg})");
            assert!((grad[[0, 0, 1]] - 1.0).abs() < 1e-12, "dc/dr1 ({nonneg})");

            // Relative-residual channel only.
            let g_c = array![[0.0]];
            let g_q = array![[1.0]];
            let grad =
                residual_em_score_vjp(x.view(), recon.view(), nonneg, g_c.view(), g_q.view());
            assert!((grad[[0, 0, 0]] - 0.0).abs() < 1e-12, "dq/dr0 ({nonneg})");
            assert!((grad[[0, 0, 1]] - (-0.8)).abs() < 1e-12, "dq/dr1 ({nonneg})");
        }
    }

    /// The non-negative code is clamped where `s < 0`, so both its value and its
    /// gradient vanish there, while the signed convention keeps the true
    /// least-squares coefficient and a live gradient. `x = [2, 1]`, `r = [−1, 0]`
    /// gives `s = −2`: nonneg `c = 0` (flat, zero grad), signed `c = −2`.
    #[test]
    fn residual_em_score_clamp_kills_gradient_but_signed_survives() {
        let x = array![[2.0, 1.0]];
        let recon =
            ndarray::Array3::from_shape_vec((1, 1, 2), vec![-1.0, 0.0]).unwrap();
        let g_c = array![[1.0]];
        let g_q = array![[1.0]];

        let (code_nn, _) = residual_em_score(x.view(), recon.view(), true);
        assert!((code_nn[[0, 0]] - 0.0).abs() < 1e-12, "clamped code is 0");
        let grad_nn =
            residual_em_score_vjp(x.view(), recon.view(), true, g_c.view(), g_q.view());
        assert!(grad_nn[[0, 0, 0]].abs() < 1e-12, "clamped grad r0 = 0");
        assert!(grad_nn[[0, 0, 1]].abs() < 1e-12, "clamped grad r1 = 0");

        let (code_sg, _) = residual_em_score(x.view(), recon.view(), false);
        assert!((code_sg[[0, 0]] - (-2.0)).abs() < 1e-12, "signed code = -2");
        let grad_sg =
            residual_em_score_vjp(x.view(), recon.view(), false, g_c.view(), g_q.view());
        // Signed coefficient is active, so the gradient is non-trivial.
        assert!(
            grad_sg[[0, 0, 0]].abs() + grad_sg[[0, 0, 1]].abs() > 1e-9,
            "signed grad is live"
        );
    }

    /// Every atom has a distinct stable label so a desync diagnostic can name
    /// the offending term.
    #[test]
    fn atoms_have_distinct_labels() {
        let crit = sample_criterion();
        let labels: Vec<&str> = crit.atoms().iter().map(SaeCriterionAtom::label).collect();
        let mut sorted = labels.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "labels must be distinct");
    }
}
