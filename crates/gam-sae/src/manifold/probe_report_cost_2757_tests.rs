#![cfg(test)]
//! #2757 probe — where the post-fit certification's wall-clock actually goes,
//! phase by phase, at the shape the issue was filed on.
//!
//! The issue measured `fit_diagnostics_report` at 3160.5 s / 45.97 GiB for
//! `p = 4096` and read a `dim ∝ p`, `time ∝ dim³`, `memory ∝ dim²` law off two
//! cells. The block-structured curvature (`2af28dddb`) removed the dense
//! `param_dim × param_dim` eigendecomposition on the branch where the metric
//! does not couple output coordinates. **This probe does not assume that
//! finished the job.** It times every phase of the report separately so the
//! surviving cost is measured rather than inferred, on both metric branches:
//!
//! * `metric.drives_gauge() == false` (Euclidean — what `diagnostic_metric`
//!   installs with no harvest, and what the #2731 cell ran) → the curvature is
//!   `p` blocks of `D × D`;
//! * `metric.drives_gauge() == true` (output-Fisher) → the curvature falls back
//!   to a root, or to a dense `(p·D)²` Gram once the root has more rows than
//!   columns, which is exactly the object #2757 is named for.
//!
//! ## Why the shapes are constants and the runs are not `#[ignore]`d
//!
//! Both stopwatches arrived (`7917759c7`) as `#[ignore]`d tests reading their
//! sweep out of `GAM_2757_*` environment variables. Each of those is a build
//! ban in this workspace — `#[ignore]` because a test that never runs is not a
//! statement, `env::var` because a run whose shape comes from the environment
//! is not reproducible from the tree — so the scanner aborted **every** build
//! in the workspace and no lane could compile anything. See `0c9ed39c5` for the
//! same lesson on the #2714 probe.
//!
//! The instrument is unchanged in what it measures; only its entry conditions
//! are. The sweep is a `const` below (raise it in a working tree to reach the
//! production cell), and the committed shape is small enough that the phase
//! table is produced on every run rather than never. Read it with
//!
//! ```sh
//! cargo test -p gam-sae --release --lib probe_2757 -- --nocapture
//! ```

