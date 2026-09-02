//! Discovery-recall control for the Fisher-mass enrichment producer
//! (`RowSamplingMeasure`, role (c) of #980).
//!
//! # The scenario this guards
//!
//! A SAE-manifold fit sees many rows. Most are *common* and *behaviorally
//! quiet*: they activate a frequent feature that barely moves the model's
//! output (low output-Fisher mass). A handful of rows are *rare* but
//! *behaviorally loud*: they carry a feature that, when present, strongly drives
//! the output (high Fisher mass). Under plain uniform sampling, the rare-loud
//! rows are drowned — a discovery/seeding batch almost never looks at them, so
//! the important feature is never surfaced. That is the recall failure the
//! Fisher-mass enrichment measure is built to fix: it OVERSAMPLES the live rows.
//!
//! # What we assert
//!
//! 1. **Under-representation baseline.** Uniform sampling gives the rare rows
//!    only their population share of attention — vanishing for a rare feature.
//! 2. **Enrichment lifts recall.** The Fisher-mass measure assigns the rare-loud
//!    rows weight far above their population share, so their expected (and
//!    actually drawn, deterministically) representation in a batch rises
//!    markedly.
//! 3. **No loss is altered.** The enrichment is a SAMPLING measure only: it is
//!    derived purely from the metric's geometry and never multiplies, reweights,
//!    or otherwise touches any per-row loss / quad-form. We assert the per-row
//!    `quad_form` (the criterion-facing residual square) is bit-for-bit
//!    identical whether or not the enrichment measure exists.
//!
//! All draws are deterministic (fixed seeds, index-derived jitter); no clock.

