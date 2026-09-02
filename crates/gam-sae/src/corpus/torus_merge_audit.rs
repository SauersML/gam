//! ISA post-fit **torus-merge audit** — the discovery-side dual of the ISA
//! birth producer (`crate::manifold::isa_seed`).
//!
//! # What it catches
//!
//! The stagewise birth path can accept two atoms that are really **one curved
//! structure** seen through two charts: a single circle whose 2-plane got split
//! across two births, or a 2-torus mistaken for two independent circles. The
//! birth producer refuses *blends* at proposal time via the fourth-moment
//! contrast, but nothing re-checks a pair of atoms **after** they are both
//! accepted and their rows are assigned. This audit does exactly that, reusing
//! the same analytic anchor.
//!
//! # The contrast (identical fourth-order anchor as the producer)
//!
//! For a pair of atoms `(a, b)` restrict to the rows where **both** are active
//! (their co-assigned rows). On that joint 2-plane form the squared radius
//! `s = y_a² + y_b²` and its normalized energy fourth moment
//!
//! ```text
//!   κ = E[s²] / E[s]²        (= E[(‖Wᵀz‖²)²] / E[‖Wᵀz‖²]²)
//! ```
//!
//! the exact quantity `isa_seed` rotates to
//! maximize. The population anchors are the producer's:
//!
//! * `κ ≈ 1` — a dense constant-radius circle (both atoms tracing one ring):
//!   strongly **sub-Gaussian**, a merge candidate;
//! * `κ = 1/q > 2` — a circle gated on a fraction `q` of the co-active rows:
//!   **super-Gaussian**, a merge candidate;
//! * `κ = 2` — a Gaussian blend of many independent charts (`s` is a scaled
//!   `χ²₂`): two genuinely independent atoms, **no** merge.
//!
//! So `(κ − 2)²` is again the contrast, and a pair is flagged **by evidence**:
//! the standardized distance `|κ̂ − 2| / SE(κ̂)` must clear the `z = 3`
//! resolution level the producer's certificate uses, and the co-active row count
//! must clear the same delta-method concentration floor. `SE(κ̂)` is the
//! plug-in delta-method standard error of the moment ratio (no autodiff, no
//! finite differences — SPEC.md §1/§2), so the flag carries a real significance,
//! not a tuned threshold.
//!
//! This module reads only a supplied per-row latent-activation matrix; it writes
//! nothing into any atom, loss, or criterion — it emits *merge candidates* for a
//! consumer to act on, exactly as the birth producer emits *proposals*.

/// A flagged "two atoms are really one curved structure" merge candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MergeCandidate {
    /// The two atom column indices (`atom_a < atom_b`).
    pub atom_a: usize,
    pub atom_b: usize,
    /// Co-active row count the contrast was estimated on.
    pub n_coactive: usize,
    /// Observed joint-plane fourth-moment `κ̂`.
    pub kappa: f64,
    /// Standardized contrast `(κ̂ − 2) / SE(κ̂)` — negative for a sub-Gaussian
    /// ring (`κ < 2`), positive for a super-Gaussian gated circle (`κ > 2`).
    pub z_score: f64,
}

impl MergeCandidate {
    /// Evidence magnitude `|z|` — how many σ the pair sits from the Gaussian
    /// blend anchor. Higher ⇒ stronger merge evidence.
    pub fn evidence(&self) -> f64 {
        self.z_score.abs()
    }
}

