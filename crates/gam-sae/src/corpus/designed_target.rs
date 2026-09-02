//! Designed corpus target collection — the #991 bridge from a streaming
//! [`CorpusRowSource`] to the in-memory row set + honesty weights the SAE term
//! fits on.
//!
//! # The architecture this realizes
//!
//! At frontier scale the fit never sees the whole corpus: it sees a **designed
//! sample** whose inclusion weights ride into the likelihood so the criterion
//! stays unbiased (#987 / #973). That makes "fit the corpus" a two-step
//! pipeline with a bounded memory footprint by construction:
//!
//! 1. **Design** — a [`RowSamplingMeasure`] over the corpus (uniform on a first
//!    harvest; [`TieredHarvest::corpus_measure`]-driven once Fisher factors
//!    exist) picks `budget` rows via
//!    [`RowSamplingMeasure::designed_subsample`] (deterministic, seeded, honest `1/π`
//!    weights).
//! 2. **Collect** — one deterministic streaming pass over the source
//!    materializes exactly those rows (the only dense `f64` block the fit ever
//!    holds: `budget × p`, not `N × p`), aligned with their weights and global
//!    `row_id`s.
//!
//! The term consumes the result as `(target, set_row_loss_weights)`; the
//! weights enter the objective through the term's single `√w` honesty seam.
//!
//! # Exactness degeneracy (the bit-identity contract)
//!
//! `budget ≥ corpus rows` (always the case below
//! [`designed_sampling_mandatory`]'s threshold unless a caller narrows it)
//! collects **every** row in stream order with weight exactly `1.0` — and the
//! term stores all-equal weights as `None`, so a shard-backed full-budget fit
//! is **bit-for-bit** the in-memory fit of the same rows. Selectivity is then
//! purely a budget decision, not a code path: drivers call this
//! unconditionally and let [`auto_designed_budget`] decide.

use ndarray::Array2;

use gam_solve::row_sampling_measure::MeasureProvenance;

/// Default designed-sample budget once [`designed_sampling_mandatory`] fires.
/// Auto-derived policy, not a knob: 2·10⁶ rows is comfortably in-memory at any
/// realistic activation width (`2e6 × 4096 × 8B ≈ 64 GiB` is the extreme; at
/// GPT-2-small widths it is ~6 GiB), large enough that designed-sample SEs on
/// shared structure are far below fit noise, and small enough that an outer
/// iteration's full pass over the *sample* is minutes, not days.
pub const DESIGNED_SAMPLE_DEFAULT_BUDGET_ROWS: usize = 2_000_000;

/// The collected designed row set: the dense fit target plus everything needed
/// to keep the fit honest and traceable back to the corpus.
#[derive(Debug, Clone)]
pub struct DesignedCorpusTarget {
    /// `(n_selected × p)` upcast activations of exactly the designed rows, in
    /// ascending global row order.
    pub target: Array2<f64>,
    /// Global corpus `row_id` of each target row (ascending). These are the
    /// keys for warm-state reuse ([`super::warm_state`]) and for aligning a
    /// [`TieredHarvest`] Fisher tier with the fitted rows.
    pub row_ids: Vec<u64>,
    /// Per-selected-row Horvitz–Thompson likelihood weight `1/π`, aligned with
    /// `target` rows. Hand to `SaeManifoldTerm::set_row_loss_weights` (which
    /// mean-normalizes; an exact full pass yields all-`1.0` here and the
    /// unweighted path there).
    pub likelihood_weights: Vec<f64>,
    /// Provenance of the measure that shaped the design.
    pub provenance: MeasureProvenance,
    /// Total corpus rows the design was drawn from.
    pub corpus_rows: u64,
}

impl DesignedCorpusTarget {
    /// Number of collected rows.
    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }

}

#[cfg(test)]
mod tests {
    use super::super::shard_reader::{MmapShardSource, encode_shard_bytes};
    use super::*;
    use ndarray::Array2 as NdArray2;
    use std::io::Write;
    use std::path::PathBuf;

    fn planted_rows(n: usize, p: usize) -> NdArray2<f64> {
        NdArray2::from_shape_fn((n, p), |(i, j)| {
            let x = (i as f64 + 1.0) * 0.7390851 + (j as f64 + 1.0) * 1.6180339;
            (x.sin() * 43_758.547).fract() * 2.0 - 1.0
        })
    }

    fn temp_shard_dir(name: &str, rows: &NdArray2<f64>, split_at: usize) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "gam-designed-target-test-{}-{}",
            std::process::id(),
            name
        ));
        std::fs::create_dir_all(&dir).expect("create dir");
        let parts = [
            ("a.shard", rows.slice(ndarray::s![..split_at, ..])),
            ("b.shard", rows.slice(ndarray::s![split_at.., ..])),
        ];
        for (key, part) in parts {
            let bytes = encode_shard_bytes(part);
            let mut f = std::fs::File::create(dir.join(key)).expect("create shard");
            f.write_all(&bytes).expect("write shard");
            f.sync_all().expect("sync");
        }
        dir
    }

    #[test]
    fn full_budget_collects_every_row_bit_for_bit_with_unit_weights() {
        let n = 137;
        let p = 5;
        let rows = planted_rows(n, p);
        let dir = temp_shard_dir("full", &rows, 60);
        let mut src = MmapShardSource::open_dir(&dir).expect("open");
        let collected = collect_designed_target_auto(&mut src, 7).expect("collect");

        assert!(!collected.is_designed_subsample());
        assert_eq!(collected.row_ids, (0..n as u64).collect::<Vec<_>>());
        assert!(collected.likelihood_weights.iter().all(|&w| w == 1.0));
        // Bit-identity to the f32-storage round-trip of the source rows: the
        // collection adds nothing on top of the shard format's own rounding.
        let stored = rows.mapv(|v| f64::from(v as f32));
        for (a, b) in collected.target.iter().zip(stored.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        if let Err(error) = std::fs::remove_dir_all(&dir) {
            log::debug!("designed-target test: removing the temp shard dir failed: {error}");
        }
    }

    #[test]
    fn designed_budget_collects_exactly_the_designed_rows_with_their_weights() {
        let n = 200;
        let p = 3;
        let rows = planted_rows(n, p);
        let dir = temp_shard_dir("designed", &rows, 90);
        let mut src = MmapShardSource::open_dir(&dir).expect("open");

        let budget = 40usize;
        let seed = 17u64;
        let collected = collect_designed_target(&mut src, None, budget, seed).expect("collect");
        assert!(collected.is_designed_subsample());

        // The selection must be the measure's own design, row for row,
        // weight for weight.
        let sample = RowSamplingMeasure::uniform(n).designed_subsample(budget, seed);
        assert_eq!(
            collected.row_ids,
            sample.rows.iter().map(|&r| r as u64).collect::<Vec<_>>()
        );
        assert_eq!(collected.likelihood_weights, sample.likelihood_weights);

        // Each collected row is bitwise the corpus row it claims to be.
        let stored = rows.mapv(|v| f64::from(v as f32));
        for (k, &rid) in collected.row_ids.iter().enumerate() {
            for c in 0..p {
                assert_eq!(
                    collected.target[[k, c]].to_bits(),
                    stored[[rid as usize, c]].to_bits(),
                    "row {rid} col {c}"
                );
            }
        }

        // Deterministic: same (measure, budget, seed) ⇒ identical collection.
        let again = collect_designed_target(&mut src, None, budget, seed).expect("collect again");
        assert_eq!(again.row_ids, collected.row_ids);
        for (a, b) in again.target.iter().zip(collected.target.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        if let Err(error) = std::fs::remove_dir_all(&dir) {
            log::debug!("designed-target test: removing the temp shard dir failed: {error}");
        }
    }

    #[test]
    fn measure_dimension_mismatch_is_rejected() {
        let rows = planted_rows(20, 2);
        let dir = temp_shard_dir("mismatch", &rows, 10);
        let mut src = MmapShardSource::open_dir(&dir).expect("open");
        let wrong = RowSamplingMeasure::uniform(7);
        let err = collect_designed_target(&mut src, Some(&wrong), 5, 1)
            .expect_err("mismatched measure must be rejected");
        assert!(err.contains("covers 7 rows"), "got: {err}");
        if let Err(error) = std::fs::remove_dir_all(&dir) {
            log::debug!("designed-target test: removing the temp shard dir failed: {error}");
        }
    }

    #[test]
    fn auto_budget_is_exact_below_threshold_and_bounded_above_it() {
        assert_eq!(auto_designed_budget(1_000), 1_000);
        assert_eq!(
            auto_designed_budget(99_999_999),
            99_999_999,
            "below the mandatory threshold the budget is the whole corpus"
        );
        assert_eq!(
            auto_designed_budget(100_000_000),
            DESIGNED_SAMPLE_DEFAULT_BUDGET_ROWS
        );
        assert_eq!(
            auto_designed_budget(u64::MAX),
            DESIGNED_SAMPLE_DEFAULT_BUDGET_ROWS
        );
    }
}
