//! #980 verification §4 / #981 Theorem-2 arm: **the functional metric is an
//! identifiability technology**.
//!
//! Rung 4 of the gauge-reduction ladder claims: with the isometry pin computed
//! in the model's own output-Fisher pullback metric, the within-atom frame
//! freedom that survives a *Euclidean* pin (`Isom(M_k)` rotations — replicate
//! fits visibly differ by them) is cut to the symmetry group of the downstream
//! readout, which is **generically trivial** — the fit is identified up to atom
//! permutation alone.
//!
//! This test realises the two arms at the certificate level with the pin root
//! derived **honestly from the metric** (not hand-fed): the isometry penalty
//! pins the pulled-back gram `G(F) = Fᵀ W F` to a reference, so its curvature
//! root along frame perturbations is the gram derivative
//!
//! ```text
//! R[(a,b), (i,c)] = δ_{cb} (W F)_{ia} + δ_{ca} (W F)_{ib}
//! ```
//!
//! — the same formula for both arms, with only `W` differing:
//!
//! * **Euclidean arm** `W = I`: for an orthonormal frame the gram derivative
//!   along the `so(2)` rotation `Ξ = F·A` is `ΞᵀF + FᵀΞ = Aᵀ + A = 0` — the
//!   rotation orbit is exactly flat, the certificate must report the rotation
//!   **unpinned** (this is "euclidean replicates disagree up to rotation").
//! * **Functional arm** `W = U Uᵀ` anisotropic (a generic readout): the same
//!   derivative is `AᵀW + WA ≠ 0` — the rotation orbit costs penalty, the
//!   certificate must report it **pinned**, leaving no residual freedom
//!   (single atom ⇒ "identified up to atom permutation" is the trivial group).
//!
//! The functional arm is also the *mixed-generator* regime the verdict rule
//! must get right: the rotation's relative curvature fraction is strictly
//! interior (= 9/128 here) — partial curvature. A rank-increase test would
//! call that a surviving freedom (under-claiming identification); the
//! relative-curvature rule must call it pinned and report the fraction.

