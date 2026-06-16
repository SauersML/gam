//! Multi-GPU device pool.
//!
//! The runtime probe (`super::runtime`) already discovers every usable CUDA
//! device into `GpuRuntime::devices` (sorted by [`GpuDeviceInfo::score`] desc),
//! but every dispatch path historically pinned its work to the single primary
//! `GpuRuntime::device`. This module turns that pool into usable parallelism:
//!
//!   * [`GpuRuntime::device_ordinals`] / [`GpuRuntime::device_count`] expose the
//!     full set of usable ordinals (highest-score first).
//!   * [`GpuRuntime::memory_budget_for`] gives a per-device byte budget so each
//!     tile can size its device buffers against the device it actually runs on.
//!   * [`balanced_partition`] splits `n` independent work items across the pool
//!     weighted by each device's [`GpuDeviceInfo::score`].
//!   * [`scatter_batched`] runs an independent-per-item closure across every
//!     device concurrently, binding each ordinal's context on its own worker
//!     thread.
//!
//! ## Concurrency model
//!
//! Per-device fan-out uses [`std::thread::scope`], **not** rayon. A rayon
//! `par_iter` worker that reaches a `OnceLock::get_or_init` whose closure itself
//! does `into_par_iter` deadlocks the whole process (team-known hazard), and the
//! cudarc context cache (`cuda_context_for`) is exactly such a lazily-initialized
//! global. Scoped OS threads sidestep that entirely: each worker calls
//! `ctx.bind_to_thread()` for its ordinal before issuing any CUDA work, so the
//! thread-local current context is correct for every kernel launched on it.

use super::device::GpuDeviceInfo;
use super::runtime::GpuRuntime;

impl GpuRuntime {
    /// Ordinals of all usable devices, highest-score first.
    ///
    /// `self.devices` is already score-sorted at probe time, so this simply
    /// projects out the ordinals. Empty only if the probe somehow produced no
    /// devices (the public `probe()` guarantees at least one on `Ok(Some(_))`).
    #[must_use]
    pub fn device_ordinals(&self) -> Vec<usize> {
        self.devices.iter().map(|device| device.ordinal).collect()
    }

    /// Number of usable devices in the pool.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Per-device byte budget: free memory capped at half of total, matching the
    /// primary-device budget computed in `runtime::probe`. Falls back to the
    /// primary `memory_budget_bytes` when the ordinal is not in the pool so a
    /// caller that passes a stale ordinal still gets a usable (conservative)
    /// budget rather than zero.
    #[must_use]
    pub fn memory_budget_for(&self, ordinal: usize) -> usize {
        self.devices
            .iter()
            .find(|device| device.ordinal == ordinal)
            .map_or(self.memory_budget_bytes, GpuDeviceInfo::memory_budget_bytes)
    }
}

/// Partition `n_units` independent work items across all usable devices,
/// weighted by [`GpuDeviceInfo::score`].
///
/// Returns `(ordinal, Range)` tiles that exactly cover `0..n_units` with no gaps
/// or overlaps, largest-score device first. A single device yields one full-span
/// tile. `n_units == 0` or no GPU yields an empty `Vec`.
///
/// Allocation is largest-remainder by score: each device's ideal share is
/// `score_i / Σscore · n_units`; floors are assigned first, then the remaining
/// units (from rounding) go to the devices with the largest fractional parts.
/// This keeps the split proportional to capability while guaranteeing the tiles
/// tile the whole range. Devices that round to a zero-width tile are dropped so
/// no worker is spawned for empty work.
#[must_use]
pub fn balanced_partition(rt: &GpuRuntime, n_units: usize) -> Vec<(usize, std::ops::Range<usize>)> {
    if n_units == 0 || rt.devices.is_empty() {
        return Vec::new();
    }
    if rt.devices.len() == 1 {
        return vec![(rt.devices[0].ordinal, 0..n_units)];
    }

    let scores: Vec<f64> = rt
        .devices
        .iter()
        .map(|device| device.score().max(0.0))
        .collect();
    let total_score: f64 = scores.iter().sum();

    // Degenerate weighting (all scores zero/non-finite): fall back to an even
    // split so we never collapse the whole batch onto one device.
    let even = !(total_score.is_finite() && total_score > 0.0);

    let n = n_units as f64;
    let mut counts: Vec<usize> = Vec::with_capacity(rt.devices.len());
    let mut remainders: Vec<(usize, f64)> = Vec::with_capacity(rt.devices.len());
    let mut assigned = 0usize;
    for (idx, score) in scores.iter().enumerate() {
        let ideal = if even {
            n / rt.devices.len() as f64
        } else {
            n * score / total_score
        };
        let floor = ideal.floor();
        let count = floor as usize;
        counts.push(count);
        assigned += count;
        remainders.push((idx, ideal - floor));
    }

    // Distribute the leftover units (from flooring) to the largest fractional
    // remainders, breaking ties toward the higher-score (earlier) device.
    let mut leftover = n_units.saturating_sub(assigned);
    if leftover > 0 {
        remainders.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        for (idx, _) in &remainders {
            if leftover == 0 {
                break;
            }
            counts[*idx] += 1;
            leftover -= 1;
        }
    }

    let mut tiles = Vec::with_capacity(rt.devices.len());
    let mut start = 0usize;
    for (idx, device) in rt.devices.iter().enumerate() {
        let count = counts[idx];
        if count == 0 {
            continue;
        }
        let end = start + count;
        tiles.push((device.ordinal, start..end));
        start = end;
    }
    assert_eq!(start, n_units, "balanced_partition tiles must cover 0..n");
    tiles
}

/// Run independent work across ALL devices concurrently.
///
/// Splits `items` via [`balanced_partition`]; each tile runs on its own
/// [`std::thread::scope`] thread that binds that ordinal's context
/// (`cuda_context_for(ordinal).bind_to_thread()`) before calling
/// `f(ordinal, &mut items[range])`. Returns `Some(())` only if EVERY tile's
/// closure returned `Some(())`; if any tile fails, panics, or a context cannot
/// be bound, returns `None` so the caller can run its deterministic whole-batch
/// CPU fallback over the (still untouched-by-a-successful-result) `items`.
///
/// Non-linux builds have no CUDA contexts to bind and so always return `None`.
#[cfg(target_os = "linux")]
#[must_use]
pub fn scatter_batched<T: Send>(
    rt: &GpuRuntime,
    items: &mut [T],
    f: impl Fn(usize, &mut [T]) -> Option<()> + Sync,
) -> Option<()> {
    let n_units = items.len();
    let tiles = balanced_partition(rt, n_units);
    if tiles.is_empty() {
        return None;
    }

    // Carve `items` into disjoint mutable sub-slices matching the tiles so each
    // worker thread owns its range exclusively (no aliasing across threads).
    let mut slices: Vec<(usize, &mut [T])> = Vec::with_capacity(tiles.len());
    let mut rest = items;
    let mut consumed = 0usize;
    for (ordinal, range) in &tiles {
        let take = range.end - consumed;
        let (head, tail) = rest.split_at_mut(take);
        slices.push((*ordinal, head));
        rest = tail;
        consumed = range.end;
    }

    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = slices
            .into_iter()
            .map(|(ordinal, slice)| {
                scope.spawn(move || {
                    // Bind this ordinal's cached context on this worker thread so
                    // every CUDA launch the closure issues targets `ordinal`.
                    let ctx = super::runtime::cuda_context_for(ordinal)?;
                    ctx.bind_to_thread().ok()?;
                    f(ordinal, slice)
                })
            })
            .collect();

        // A panicking worker yields `Err` from `join`; treat it like a tile
        // failure so the caller falls back to CPU for the whole batch.
        let mut all_ok = true;
        for handle in handles {
            match handle.join() {
                Ok(Some(())) => {}
                _ => all_ok = false,
            }
        }
        if all_ok { Some(()) } else { None }
    })
}

/// Non-linux `scatter_batched`: there are no CUDA contexts to bind off Linux,
/// so device fan-out is unavailable.
///
/// This must exist on every target, not just Linux: not every caller is inside
/// `#[cfg(target_os = "linux")]` — the SAE manifold per-atom Gram/smoothness
/// scatters (`src/terms/sae/manifold/mod.rs`) call it from platform-independent
/// code. At runtime off Linux `GpuRuntime::global()` returns `None`, so the
/// `Some(rt)` branch that reaches here is never taken; the body only needs to
/// compile and honour the contract. `balanced_partition` yields no tiles when
/// the runtime has no devices, so this reports `None` and the caller runs its
/// deterministic whole-batch CPU fallback. The per-tile invocation is kept so
/// the contract is honoured verbatim if a non-Linux backend ever exposes
/// devices: each tile's closure runs over its own disjoint sub-slice, with no
/// device binding to perform on this platform (the only step the Linux path
/// adds).
#[cfg(not(target_os = "linux"))]
#[must_use]
pub fn scatter_batched<T: Send>(
    rt: &GpuRuntime,
    items: &mut [T],
    f: impl Fn(usize, &mut [T]) -> Option<()> + Sync,
) -> Option<()> {
    let tiles = balanced_partition(rt, items.len());
    if tiles.is_empty() {
        return None;
    }
    let mut rest = items;
    let mut consumed = 0usize;
    let mut all_ok = true;
    for (ordinal, range) in &tiles {
        let take = range.end - consumed;
        let (head, tail) = rest.split_at_mut(take);
        if f(*ordinal, head).is_none() {
            all_ok = false;
        }
        rest = tail;
        consumed = range.end;
    }
    if all_ok { Some(()) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::device::{GpuCapability, GpuDeviceInfo};
    use crate::gpu::policy::GpuDispatchPolicy;

    fn device_with(ordinal: usize, sm_count: i32, free_gib: f64) -> GpuDeviceInfo {
        GpuDeviceInfo {
            ordinal,
            name: format!("synthetic-{ordinal}"),
            capability: GpuCapability::from_compute_capability(7, 0),
            sm_count,
            max_threads_per_sm: 2048,
            max_shared_mem_per_block: 49_152,
            l2_cache_bytes: 6 * 1024 * 1024,
            total_mem_bytes: (free_gib as usize) * 1_073_741_824 * 2,
            free_mem_bytes: (free_gib * 1_073_741_824.0) as usize,
            ecc_enabled: false,
            integrated: false,
            mig_mode: false,
        }
    }

    fn runtime_with(devices: Vec<GpuDeviceInfo>) -> GpuRuntime {
        let device = devices
            .first()
            .cloned()
            .expect("test runtime needs ≥1 device");
        let memory_budget_bytes = device.free_mem_bytes.min(device.total_mem_bytes / 2);
        GpuRuntime {
            device,
            devices,
            policy: GpuDispatchPolicy::default(),
            memory_budget_bytes,
        }
    }

    /// Tiles must exactly tile `0..n_units`: contiguous, gap-free, no overlap,
    /// and ordered largest-score-device first.
    fn assert_covers(tiles: &[(usize, std::ops::Range<usize>)], n_units: usize) {
        let mut cursor = 0usize;
        for (_, range) in tiles {
            assert_eq!(range.start, cursor, "tile gap/overlap at {cursor}");
            assert!(range.end > range.start, "empty tile emitted");
            cursor = range.end;
        }
        assert_eq!(cursor, n_units, "tiles must cover the whole range");
    }

    #[test]
    fn single_device_one_full_tile() {
        let rt = runtime_with(vec![device_with(0, 80, 16.0)]);
        let tiles = balanced_partition(&rt, 100);
        assert_eq!(tiles, vec![(0, 0..100)]);
    }

    #[test]
    fn three_devices_even_split_when_scores_equal() {
        // Identical devices → identical scores → even split (largest-remainder
        // pushes the +1 toward the earliest devices for the rounding leftover).
        let rt = runtime_with(vec![
            device_with(0, 80, 16.0),
            device_with(1, 80, 16.0),
            device_with(2, 80, 16.0),
        ]);
        let tiles = balanced_partition(&rt, 99);
        assert_eq!(
            tiles,
            vec![(0, 0..33), (1, 33..66), (2, 66..99)],
            "equal scores must split evenly"
        );
        assert_covers(&tiles, 99);

        // 100 units across 3 equal devices: 34/33/33, extra unit to the first.
        let tiles = balanced_partition(&rt, 100);
        assert_eq!(tiles, vec![(0, 0..34), (1, 34..67), (2, 67..100)]);
        assert_covers(&tiles, 100);
    }

    #[test]
    fn three_devices_weighted_by_unequal_scores() {
        // Device 0 has far more SMs and memory than 1 and 2, so it must take a
        // strictly larger tile; the split stays proportional and tiling holds.
        let devices = vec![
            device_with(0, 132, 40.0),
            device_with(1, 40, 8.0),
            device_with(2, 40, 8.0),
        ];
        let rt = runtime_with(devices.clone());
        let n_units = 1000;
        let tiles = balanced_partition(&rt, n_units);
        assert_covers(&tiles, n_units);
        // Highest-score device first and its tile is the largest.
        assert_eq!(tiles[0].0, 0);
        let widths: Vec<usize> = tiles.iter().map(|(_, r)| r.end - r.start).collect();
        assert!(
            widths[0] > widths[1] && widths[0] > widths[2],
            "highest-score device must get the largest tile, got {widths:?}"
        );
        // Tiles 1 and 2 have equal scores → equal widths.
        assert_eq!(widths[1], widths[2]);
        // Proportionality: each width tracks score share within ±1 unit.
        let total_score: f64 = devices.iter().map(GpuDeviceInfo::score).sum();
        for (device, width) in devices.iter().zip(&widths) {
            let ideal = device.score() / total_score * n_units as f64;
            assert!(
                (*width as f64 - ideal).abs() <= 1.0,
                "width {width} not within 1 of ideal {ideal} for ordinal {}",
                device.ordinal
            );
        }
    }

    #[test]
    fn fewer_units_than_devices_drops_empty_tiles() {
        // 2 units, 5 devices: only the 2 highest-score devices get a tile and
        // no zero-width tile is emitted.
        let rt = runtime_with(vec![
            device_with(0, 132, 40.0),
            device_with(1, 100, 24.0),
            device_with(2, 80, 16.0),
            device_with(3, 60, 12.0),
            device_with(4, 40, 8.0),
        ]);
        let tiles = balanced_partition(&rt, 2);
        assert_covers(&tiles, 2);
        assert_eq!(tiles.len(), 2, "one tile per unit when units < devices");
        assert_eq!(tiles[0].0, 0, "highest-score device served first");
        assert_eq!(tiles[1].0, 1);
    }

    #[test]
    fn zero_units_yields_no_tiles() {
        let rt = runtime_with(vec![device_with(0, 80, 16.0), device_with(1, 80, 16.0)]);
        assert!(balanced_partition(&rt, 0).is_empty());
    }

    #[test]
    fn device_ordinals_and_count_track_pool() {
        let rt = runtime_with(vec![
            device_with(0, 80, 16.0),
            device_with(3, 80, 16.0),
            device_with(5, 80, 16.0),
        ]);
        assert_eq!(rt.device_count(), 3);
        assert_eq!(rt.device_ordinals(), vec![0, 3, 5]);
    }

    #[test]
    fn memory_budget_for_caps_free_at_half_total() {
        // free = 8 GiB, total = 16 GiB → budget = min(8, 8) = 8 GiB.
        let rt = runtime_with(vec![device_with(0, 80, 8.0)]);
        let gib = 1_073_741_824usize;
        assert_eq!(rt.memory_budget_for(0), 8 * gib);
        // Unknown ordinal falls back to the primary budget rather than zero.
        assert_eq!(rt.memory_budget_for(99), rt.memory_budget_bytes);
    }
}
