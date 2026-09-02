//! Fit-level verification battery for the #972 Grassmann decoder frames
//! (issue #992, items 2 and 3).
//!
//! * **Evidence equality at small p** — a dictionary of truly low-rank atoms
//!   evaluated through the SAME criterion entry (`penalized_quasi_laplace_criterion`) twice: once
//!   on the full-`B` border, once with the Grassmann frames activated. Frame
//!   activation is an exact re-representation (the decoder matrix is
//!   unchanged; only the border coordinates and the Laplace dimension
//!   accounting move), so the criterion must agree within a small relative
//!   tolerance — this is the test that catches a drift in
//!   `grassmann_evidence_dimension()` → `reml_occam_term` (the profiled-frame
//!   normalizer), the objective↔gradient-desync class wearing evidence
//!   clothing. The border-size collapse is asserted as a hard invariant
//!   alongside (`factored_border_dim == Σ M_k·r_k < beta_dim`).
//!
//! * **Alternating-update stationarity** — the closed-form streaming polar
//!   update is exact block-coordinate ascent for the frame: at an alternating
//!   `(U, C)` fixed point of the factored least-squares objective the joint
//!   gradient restricted to the Stiefel tangent space at `U` must vanish.
//!   Probed by central finite differences along a basis of tangent directions
//!   (both the `U·A` antisymmetric leg and the `U_⊥·B` normal leg), with the
//!   in-tree `GrassmannCrossMoment` / `GrassmannFrame::polar_update`
//!   primitives driving the alternation — the same code path the term's
//!   frame refresh uses.

// ---------------------------------------------------------------------------
// Designed-subsample honesty at the fit level (#991 acceptance): a fit on a
// non-uniformly designed, HT-weighted subsample must recover the same planted
// structure as the full fit.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Item 3: alternating polar/LS stationarity (FD against the joint gradient).
// ---------------------------------------------------------------------------

