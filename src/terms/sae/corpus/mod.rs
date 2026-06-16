//! Streaming / out-of-core corpus driver for the SAE term (#973).
//!
//! This module is the **driver** that lets a sparse-autoencoder term fit on a
//! corpus of activations far larger than RAM, without ever materializing the
//! activations as a dense `f64` matrix. It sits *behind* the SAE term: the
//! term consumes the seam this module exposes, this module owns the streaming,
//! warm-state, scheduling and mixed-precision machinery.
//!
//! # The four pieces
//!
//! * [`shard_reader`] — an mmap-backed, bounded-prefetch reader over one or
//!   many on-disk activation shards. It defines the `v1` shard format
//!   (fixed 32-byte header + row-major `f32` payload) and yields `f64`-upcast
//!   row batches in a **deterministic global order** with stable `row_id`s,
//!   independent of OS readahead.
//! * [`warm_state`] — a disk-backed (mmap/LRU) per-row warm-state cache keyed
//!   by `(row_id, TermCollectionSpec structural hash)`. It persists each row's
//!   inner-solve seed (latent coords + active set) so re-solving the same row
//!   across outer ρ passes (or across a SIGKILL-resume) costs ~3 inner
//!   iterations instead of ~30. The structural hash is computed the same way
//!   the existing warm-start cache does (#869,
//!   `TermCollectionSpec::write_structural_shape_hash`), so distinct topologies
//!   never cross-seed.
//! * [`rho_cascade`] — a subsample-converge-then-full-pass ρ schedule that
//!   carries **importance weights**. Every outer ρ step is a full corpus pass
//!   in expectation; early steps run on a deterministic hashed-`row_id`
//!   subsample with each included row reweighted by `1/inclusion_probability`
//!   (the subsample-honesty contract), and the trailing steps are honest full
//!   passes.
//! * [`kernels`] — fused mixed-precision kernels (`dot`, `gram`, `gemv`,
//!   `gemv_t`, `cross`) that **read `f32` rows and accumulate in `f64`**, the
//!   numerical contract that keeps the streaming sums deterministic and
//!   precise despite `f32` on-disk storage.
//! * [`object_store`] (#987) — the same `v1` shards streamed out of **object
//!   storage** through a two-method [`object_store::ObjectStore`] trait (no
//!   cloud SDK in-tree), with a bounded prefetch window and the identical
//!   deterministic `(row_id, row)` sequence as the mmap reader. Also carries
//!   the frontier predicate
//!   [`object_store::designed_sampling_mandatory`]: above 10⁸ rows the fit
//!   must see a designed, honesty-weighted subsample
//!   ([`crate::inference::row_measure::EnrichmentRowMeasure::designed_subsample`]), not
//!   a full exact pass.
//! * [`designed_target`] (#991) — the consumer bridge: design a row sample
//!   from a [`crate::inference::row_measure::EnrichmentRowMeasure`] (uniform cold start,
//!   or a previous harvest's lifted Fisher measure), stream-collect exactly
//!   those rows into the `budget × p` fit target, and hand the `1/π` honesty
//!   weights to `SaeManifoldTerm::set_row_loss_weights`. Full budget ⇒ the
//!   exact pass, bit-for-bit (weights collapse to the unweighted path).
//!
//! # The seam
//!
//! The SAE term (owned by another track, `sae_manifold.rs`) consumes exactly
//! two traits from here — re-exported below as the public seam:
//!
//! * [`CorpusRowSource`] — "give me the next deterministic batch of rows /
//!   rewind for the next ρ pass", and
//! * [`RowWarmCache`] — "give me / take back this row's inner-solve warm
//!   start".
//!
//! Together with [`rho_cascade::RhoCascadeSchedule`] (which step's subsample +
//! importance weights to apply) and the [`kernels`] (how to accumulate a
//! batch's contribution), these let the term run a full streaming, warm-started,
//! mixed-precision REML fit over an out-of-core corpus while keeping the
//! determinism and crash-resume guarantees the rest of #973 established.
//!
//! Nothing in this module references `sae_manifold.rs`; the term wires these
//! pieces in on its side of the seam.

pub mod designed_target;
pub mod kernels;
pub mod ledger_store;
pub mod object_store;
pub mod rho_cascade;
pub mod shard_reader;
pub mod warm_state;

// ---------------------------------------------------------------------------
// The driver seam consumed by the SAE term.
// ---------------------------------------------------------------------------

/// Deterministic, restartable source of activation row batches (seam half 1).
pub use shard_reader::{
    CorpusRowSource, DTYPE_F32, HEADER_LEN, MmapShardSource, RowBatch, SHARD_MAGIC, ShardError,
    encode_shard_bytes,
};

/// Per-row inner-solve warm-state cache (seam half 2).
pub use warm_state::{DiskRowWarmCache, RowWarmCache, RowWarmState};

/// Subsample → full-pass ρ schedule with importance weights.
pub use rho_cascade::{RhoCascadeSchedule, RhoStepPlan, row_in_fraction};

/// Designed corpus target collection (#991): stream → designed sample +
/// honesty weights, the row set the term actually fits.
pub use designed_target::{
    DESIGNED_SAMPLE_DEFAULT_BUDGET_ROWS, DesignedCorpusTarget, auto_designed_budget,
    collect_designed_target, collect_designed_target_auto, collect_designed_target_from_harvest,
};

/// Mixed-precision fused kernels (read `f32`, accumulate `f64`).
pub use kernels::{
    axpy_f32_into_f64, cross_f32_rows_f64, dot_f32_f64, gemv_f32_rows_f64, gemv_t_f32_rows_f64,
    gram_f32_rows_f64, norm_sq_f32_f64,
};
