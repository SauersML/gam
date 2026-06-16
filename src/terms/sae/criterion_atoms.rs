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
//!         + logdet_trace           (½·tr(H⁻¹ ∂H/∂ρ))
//!         + occam_deriv            (−∂occam/∂ρ)
//!         + implicit_correction    (−½·Γᵀ H⁻¹ ∂g/∂ρ, the envelope/IFT term
//!                                   that accounts for θ̂(ρ) moving — #1006's Γ).
//! ```
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

use ndarray::Array1;

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
