//! #1115 + #1103 end-to-end: the per-atom decoder-functional POINT summaries and
//! the any-n-valid atom-smooth structure e-value, both consumed through the
//! public `dictionary_report` surface.
//!
//! #1115 removed the #1099 per-atom curvature *confidence interval* and the
//! influence-function SE on the #1097 functionals: those conditioned on the
//! fitted latent coordinates / assignment (generated regressors estimated from
//! the same activations forming the response) as if known, so they omit the
//! generated-regressor variance channel and under-cover. What survives is the
//! penalty-debiased POINT summary (`AtomFunctionalEstimate`: plug-in +
//! debiased + removed bias, NO se/ci) — this test asserts the report exposes no
//! coverage-claiming fields.
//!
//! #1103 reports the split-likelihood-ratio e-value for "the atom's smooth is
//! non-constant" (the same universal-inference instrument the atom-birth gate
//! uses), replacing the earlier Lawley–Bartlett-corrected χ². It IS finite-
//! sample valid with no regularity conditions, so it is kept.
//!
//! The atom's inner decoder smooth is a Gaussian-identity penalized WLS fit
//! `g_k(t) = Φ_k(t)ᵀ β` with roughness Gram `S`. We assert the reports are
//! actually POPULATED (not the `None` stub) with finite point summaries and
//! positive non-constant evidence for a curved atom.

