//! [`Certificate`] implementations for the existing certificate zoo (task #16).
//!
//! The `impl Certificate for …` blocks that once lived here were relocated into
//! their types' home crates to satisfy the coherence orphan rule (task #1521):
//! the [`Certificate`](crate::inference::certificates::Certificate) trait now
//! lives in the neutral `gam-problem` crate, so each impl must be defined where
//! the implemented type is owned. The gam-solve-owned certificates
//! (`CriterionCertificate`, `CoresetCertificate`, `LogdetEnclosure`,
//! `CollapseEvent`) carry their impls in
//! [`gam_solve::inference::certificate_impls`]; the gam-sae-owned certificates
//! (`EncodeResult`, `ResidualGaugeReport`, `CertificateInputs`) carry theirs in
//! `gam_sae::certificate_impls`. The bodies moved byte-identically, so there
//! remains exactly one source of truth for each verdict.
//!
//! The two margin-resolution helpers (`coreset_race_verdict`,
//! `enclosure_margin_verdict`) descended into `gam-solve` alongside the
//! gam-solve impls; they are re-exported here so existing
//! `crate::inference::certificate_impls::*` paths resolve unchanged.

pub use gam_solve::inference::certificate_impls::{coreset_race_verdict, enclosure_margin_verdict};
