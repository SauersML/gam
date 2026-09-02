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

use super::tests_frame_curvature_2757::{
    planted_term_for_probe, source_root_rows, source_stored_scalars, source_structure_tag,
    unit_rho_for_probe,
};
use crate::identifiability::FrameColumnLayout;
use crate::manifold::construction::ResidualGaugeCurvatureSource;
use crate::manifold::streamed_frame_curvature::StreamedFrameCurvatureOperator;
use ndarray::Array2;
use std::time::Instant;

/// Rows in the committed sweep.
const PROBE_ROWS: usize = 64;
/// Charts in the committed Euclidean sweep.
const PROBE_EUCLIDEAN_CHARTS: usize = 8;
/// Output widths in the committed Euclidean sweep. The #2731 production cell is
/// `p = 2048, charts = 32, n = 256`; raise this to walk toward it.
const PROBE_EUCLIDEAN_WIDTHS: [usize; 3] = [16, 32, 64];
/// Charts in the committed gauge-driving sweep.
const PROBE_GAUGE_CHARTS: usize = 4;
/// Rank of the output-Fisher metric root in the committed gauge-driving sweep.
const PROBE_GAUGE_METRIC_RANK: usize = 2;
/// Output widths in the committed gauge-driving sweep.
const PROBE_GAUGE_WIDTHS: [usize; 3] = [8, 16, 32];

/// Rows in the gauge-branch cost-law sweep.
const PROBE_LAW_ROWS: usize = 64;
/// Charts in the gauge-branch cost-law sweep.
const PROBE_LAW_CHARTS: usize = 8;
/// Metric root rank in the gauge-branch cost-law sweep. `root_rows = n · rank`
/// is held FIXED across the sweep so the only thing that moves is `param_dim`,
/// which is what makes the fitted exponents below statements about `param_dim`
/// and not about the row count. It is also the smallest rank that keeps every
/// cell on the branch under test (`root_rows > param_dim` at the widest cell).
const PROBE_LAW_METRIC_RANK: usize = 9;
/// Output widths in the gauge-branch cost-law sweep. `param_dim = 8·p` runs
/// `128 → 512`, a factor of 4, over which a cubic moves 64x and a linear pass
/// moves 4x. The #2731 production cell is `p = 2048, charts = 32` at
/// `param_dim = 65 536`; raise these to walk toward it.
const PROBE_LAW_WIDTHS: [usize; 4] = [16, 32, 48, 64];

