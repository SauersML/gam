//! The reduced `K x K` shared-system solve: dense Schur assembly (direct and
//! square-root BA), the Schur matvec, the Jacobi/cluster/Schwarz
//! preconditioners, Steihaug-PCG, and the [`ArrowSchurError`] type.

use super::*;

/// Reduce one contiguous device tile's rows into a private `-Σ leftᵀ·right`
/// partial (`k×k`).
///
/// The tile stacks its per-row `left_i` / `right_i` factors (each `d×k`) into
/// two `(Σ_i d_i × k)` matrices and tries a single per-ordinal `AᵀB` device
/// GEMM (`gam_gpu::try_fast_atb_on_ordinal`), which runs on the device this
/// worker thread already bound — one big GPU GEMM per tile rather than `n` small
/// CPU ones. When the device primitive declines (no GPU, shape below policy,
/// transient failure) the tile reduces with the exact CPU `block_gemm_subtract`
/// loop, so the result is unchanged. The partial is negated so the caller's
/// `schur += partial` reproduces the serial `schur -= Σ contribution`.
pub(crate) fn tile_schur_partial<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
    kind: SchurReductionKind,
    ordinal: usize,
    range: Range<usize>,
) -> Result<Array2<f64>, ArrowSchurError> {
    let k = sys.k;

    // Build the per-row contribution factors once; both the GPU stacked-GEMM
    // and the CPU fallback consume them.
    let mut factors: Vec<(Array2<f64>, Array2<f64>)> = Vec::with_capacity(range.len());
    let mut total_d = 0usize;
    for i in range.clone() {
        let (left, right) = row_schur_contribution_factors(
            sys,
            i,
            &sys.rows[i],
            htt_factors.factor(i),
            backend,
            kind,
        )?;
        total_d += left.nrows();
        factors.push((left, right));
    }

    // Stack into (total_d × k) left/right matrices for one device AᵀB GEMM on
    // this tile's bound ordinal. `try_fast_atb_on_ordinal` returns leftᵀ·right
    // (k×k); negate into the partial. At an SAE-shaped whole-fit tile with
    // n=2000 rows, k=2048 shared columns, M=12 local rows per observation, and
    // K=8 candidate/atom batches, the stacked GEMM is
    // 2*(n*M)*k^2 = 201_326_592_000 flops per batch, or
    // 1_610_612_736_000 flops across K=8, so the policy work gate is cleared
    // even though the observation count is far below the old row floor.
    if total_d > 0 && k > 0 {
        let mut left_stack = Array2::<f64>::zeros((total_d, k));
        let mut right_stack = Array2::<f64>::zeros((total_d, k));
        let mut base = 0usize;
        for (left, right) in &factors {
            let di = left.nrows();
            left_stack
                .slice_mut(ndarray::s![base..base + di, ..])
                .assign(left);
            right_stack
                .slice_mut(ndarray::s![base..base + di, ..])
                .assign(right);
            base += di;
        }
        if let Some(product) =
            gam_gpu::try_fast_atb_on_ordinal(ordinal, left_stack.view(), right_stack.view())
        {
            return Ok(product.mapv(|v| -v));
        }
    }

    // CPU fallback: exact per-row block_gemm_subtract into a zero-seeded partial.
    let mut partial = Array2::<f64>::zeros((k, k));
    for (left, right) in &factors {
        backend.block_gemm_subtract(&mut partial, left, right);
    }
    Ok(partial)
}

/// A reduced-Schur chunk partial over the `(a, b)` pairs its rows touch, each
/// accumulating `-Σ_rows Σ_c left[c, a]·right[c, b]` in a dense `k×k` value array.
///
/// Every touched value is accumulated in the same row, `c`, `a`, `b` order and from
/// the same `+0.0` start as a dense zero-seeded `k×k` partial under the CPU
/// `block_gemm_subtract`, so it is the same f64. The fold adds each touched pair
/// once, and distinct pairs are distinct Schur entries, so the order the pairs are
/// visited in cannot move a word. A pair no row of the chunk touches is `+0.0` in
/// the dense zero-seeded partial and never folded here, which changes at most the
/// sign of a zero entry after the fold.
///
/// Which pairs are touched is a `k`-bit set per left index `a`. A factor row `c`
/// touches `left_active(c) × right_active(c)`, so it ORs the words of
/// `right_active(c)`'s column set into the set of every `a` in `left_active(c)`:
/// `|left_active|·|right words|` word ORs outside the product loop. Marking each
/// product instead (a touched flag and a key push per `a·k + b`) cost the
/// `inner_fit_core_scaling` system (`N = 40000`, `d = 2`, `k = 256`) 1.19 s per
/// parallel solve against 0.78 s for the unmarked dense partials it replaced
/// (sw4i 1255679).
struct TouchedPairPartial {
    k: usize,
    /// `⌈k/64⌉`, the words of one left index's touched set.
    words: usize,
    values: Vec<f64>,
    /// `touched[a·words + w]` holds columns `64·w ..` of `a`'s touched set.
    touched: Vec<u64>,
    /// The left indices with a touched pair, in first-touch order.
    touched_rows: Vec<usize>,
    row_listed: Vec<bool>,
    left_active: Vec<(usize, f64)>,
    right_active: Vec<(usize, f64)>,
    /// The nonzero words of `right_active`'s column set, `(word, bits)` in word order.
    right_words: Vec<(usize, u64)>,
}

impl TouchedPairPartial {
    fn new(k: usize) -> Self {
        let words = k.div_ceil(u64::BITS as usize);
        Self {
            k,
            words,
            values: vec![0.0; k * k],
            touched: vec![0; k * words],
            touched_rows: Vec::new(),
            row_listed: vec![false; k],
            left_active: Vec::new(),
            right_active: Vec::new(),
            right_words: Vec::new(),
        }
    }

    /// Back to every pair at `+0.0` and untouched, resetting only the touched pairs.
    fn clear(&mut self) {
        let Self {
            k,
            words,
            values,
            touched,
            touched_rows,
            row_listed,
            ..
        } = self;
        for &a in touched_rows.iter() {
            for (w, word) in touched[a * *words..(a + 1) * *words].iter_mut().enumerate() {
                let mut bits = *word;
                while bits != 0 {
                    let b = w * u64::BITS as usize + bits.trailing_zeros() as usize;
                    values[a * *k + b] = 0.0;
                    bits &= bits - 1;
                }
                *word = 0;
            }
            row_listed[a] = false;
        }
        touched_rows.clear();
    }

    /// `schur[a, b] += partial[a, b]` for every touched pair.
    fn fold_into(&self, schur: &mut Array2<f64>) {
        let schur_flat = schur
            .as_slice_mut()
            .expect("TouchedPairPartial::fold_into: reduced Schur must be standard-layout");
        for &a in &self.touched_rows {
            let row = self.touched[a * self.words..(a + 1) * self.words].iter();
            for (w, &word) in row.enumerate() {
                let mut bits = word;
                while bits != 0 {
                    let b = w * u64::BITS as usize + bits.trailing_zeros() as usize;
                    schur_flat[a * self.k + b] += self.values[a * self.k + b];
                    bits &= bits - 1;
                }
            }
        }
    }
}

/// The governor charge of one [`TouchedPairPartial`] at border `k`: its value array
/// (`k²·8` bytes), its touched sets (`k·⌈k/64⌉·8`) and its per-left-index list and
/// flag (`k·(8 + 1)`). `None` when the byte count overflows.
///
/// The per-factor-row scratch (`left_active`, `right_active`, `right_words`, at most
/// `k` entries each) is not charged.
fn touched_pair_partial_bytes(k: usize) -> Option<usize> {
    let words = k.div_ceil(u64::BITS as usize);
    let values = k.checked_mul(k)?.checked_mul(std::mem::size_of::<f64>())?;
    let touched = k.checked_mul(words)?.checked_mul(std::mem::size_of::<u64>())?;
    let rows = k.checked_mul(std::mem::size_of::<usize>() + std::mem::size_of::<bool>())?;
    values.checked_add(touched)?.checked_add(rows)
}

/// How the reduced-Schur fold runs once the memory governor has priced its partials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TouchedPairFold {
    /// Fixed row chunks reduced into this many live partials, folded in chunk order.
    /// The chunk sums and their fold order do not depend on the count, so every count
    /// folds the same words; the count only sets how many chunks reduce at once.
    Chunked { partials: usize },
    /// No partial is admitted: the rows reduce in place, serially, in row order.
    InPlace,
}

/// The most partials, up to `wanted`, whose [`touched_pair_partial_bytes`] charge the
/// ledger takes, with the reservation, or [`TouchedPairFold::InPlace`] when it takes
/// none (or the byte count overflows).
///
/// A declined footprint degrades the fold's parallelism, never its per-product cost:
/// every admitted partial is the same dense store. `remaining` is the ledger's
/// admissible bytes and `reserve` charges it. A charge refused because a peer
/// reserved first is retried at the count that remains, so the count strictly
/// decreases until one is taken or none fits.
pub(crate) fn plan_touched_pair_fold<R>(
    k: usize,
    wanted: usize,
    remaining: impl Fn() -> usize,
    mut reserve: impl FnMut(usize) -> Option<R>,
) -> (TouchedPairFold, Option<R>) {
    let Some(per_partial) = touched_pair_partial_bytes(k) else {
        return (TouchedPairFold::InPlace, None);
    };
    let admissible = |bytes: usize| bytes / per_partial.max(1);
    let mut partials = wanted.min(admissible(remaining()));
    while partials > 0 {
        if let Some(reservation) = reserve(partials * per_partial) {
            return (TouchedPairFold::Chunked { partials }, Some(reservation));
        }
        partials = (partials - 1).min(admissible(remaining()));
    }
    (TouchedPairFold::InPlace, None)
}

/// Subtract one row's Schur contribution into a [`TouchedPairPartial`].
///
/// The loop is `block_gemm_subtract`'s on the CPU backend: for each factor row `c`,
/// the nonzero entries of `left[c, ·]` and `right[c, ·]` in column order, then
/// `partial[a, b] -= l·r` over their product in `a`, `b` order. After each `a`'s
/// products, `right[c, ·]`'s column words are ORed into `a`'s touched set.
fn subtract_row_schur_contribution_touched_pairs<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    row_idx: usize,
    row: &ArrowRowBlock,
    htt_factor: ArrayView2<'_, f64>,
    backend: &B,
    kind: SchurReductionKind,
    partial: &mut TouchedPairPartial,
) -> Result<(), ArrowSchurError> {
    let (left, right) =
        row_schur_contribution_factors(sys, row_idx, row, htt_factor, backend, kind)?;
    let TouchedPairPartial {
        k,
        words,
        values,
        touched,
        touched_rows,
        row_listed,
        left_active,
        right_active,
        right_words,
    } = partial;
    let (k, words) = (*k, *words);
    for c in 0..left.nrows() {
        left_active.clear();
        right_active.clear();
        right_words.clear();
        let left_row = left.row(c);
        let right_row = right.row(c);
        let left_row = left_row
            .as_slice()
            .expect("touched-pair Schur subtract: left row must be contiguous");
        let right_row = right_row
            .as_slice()
            .expect("touched-pair Schur subtract: right row must be contiguous");
        for col in 0..k {
            let l = left_row[col];
            let r = right_row[col];
            if l != 0.0 {
                left_active.push((col, l));
            }
            if r != 0.0 {
                right_active.push((col, r));
                let word = col / u64::BITS as usize;
                let bit = 1_u64 << (col % u64::BITS as usize);
                match right_words.last_mut() {
                    Some((last, bits)) if *last == word => *bits |= bit,
                    _ => right_words.push((word, bit)),
                }
            }
        }
        if right_active.is_empty() {
            continue;
        }
        for &(a, lca) in left_active.iter() {
            let partial_row = &mut values[a * k..(a + 1) * k];
            for &(b, rcb) in right_active.iter() {
                partial_row[b] -= lca * rcb;
            }
            let row_touched = &mut touched[a * words..(a + 1) * words];
            for &(word, bits) in right_words.iter() {
                row_touched[word] |= bits;
            }
            if !row_listed[a] {
                row_listed[a] = true;
                touched_rows.push(a);
            }
        }
    }
    Ok(())
}

/// The parallel reduced-Schur fold: rows in fixed [`SCHUR_FOLD_ROW_CHUNK`]-row
/// chunks, each reduced in row order into a touched-pair partial, `partials` of them
/// at once, the partials folded into `schur` in chunk order. Bit-identical run to run
/// and at every `partials`.
pub(crate) fn fold_touched_pair_chunk_partials<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
    kind: SchurReductionKind,
    schur: &mut Array2<f64>,
    partials: usize,
) -> Result<(), ArrowSchurError> {
    let k = sys.k;
    fold_row_chunk_partials_at_width(
        sys.rows.len(),
        partials,
        || TouchedPairPartial::new(k),
        TouchedPairPartial::clear,
        |i, partial| {
            subtract_row_schur_contribution_touched_pairs(
                sys,
                i,
                &sys.rows[i],
                htt_factors.factor(i),
                backend,
                kind,
                partial,
            )
        },
        |partial| partial.fold_into(schur),
    )
}

/// Reduce the per-row Schur contributions `Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)`
/// out of `schur` (seeded with `H_ββ + ρ_β·I`).
///
/// The per-row contributions are independent — exactly the "sum over independent
/// arrow-tip blocks" axis the device pool partitions. When more than one GPU is
/// usable, [`gam_gpu::pool::balanced_partition`] splits the `0..n` rows into
/// per-device contiguous tiles; each tile is reduced on its own scoped thread
/// (binding that ordinal's context so the per-row GEMM-subtract offloads to its
/// device) into a private `k×k` partial, and the partials are summed back into
/// `schur` in tile order. The tiles are contiguous, ordered to cover `0..n`, and
/// folded back in that same order, so within each tile the per-row accumulation
/// order is preserved and the only departure from the serial loop is the
/// inter-tile reassociation of the reduction sum — the established
/// reduction-order equivalence the device pool already operates under, well
/// inside the Newton solve's tolerance.
///
/// With a single device (or no GPU) the row loop runs serially in place, which
/// is bit-for-bit the original behaviour.
pub(crate) fn reduce_row_schur_contributions<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &B,
    kind: SchurReductionKind,
    schur: &mut Array2<f64>,
    gpu_policy: gam_gpu::GpuPolicy,
) -> Result<(), ArrowSchurError> {
    let n = sys.rows.len();
    let k = sys.k;

    // Size gate BEFORE the device probe (startup-tax ordering fix): the
    // multi-GPU tile path exists to overlap the per-row `leftᵀ·right` GEMMs
    // (≈ `2·d·k²` flops each, `2·n·d·k²` total) across the pool, and each
    // tile's GEMMs still pass through the policy-gated dispatch shims — which
    // refuse every op when the WHOLE assembly is below
    // `MIN_CALIBRATABLE_GEMM_FLOPS`, the smallest floor any reachable policy
    // can carry. Such a shape would only inherit the tile split's inter-tile
    // reassociation (the documented, tolerance-bounded departure) while doing
    // 100% CPU work, so route it to the serial/rayon reference path below
    // WITHOUT resolving GPU availability (whose first call creates a CUDA
    // primary context on every GPU). Shapes clearing the floor probe and tile
    // exactly as before.
    let assembly_work = 2u128 * (n as u128) * (sys.d as u128) * (k as u128) * (k as u128);
    let tiles = if assembly_work < gam_gpu::GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS {
        None
    } else {
        gam_gpu::device_runtime::GpuRuntime::resolve(gpu_policy)
            .map_err(|error| ArrowSchurError::SchurFactorFailed {
                reason: format!("GPU runtime resolution failed during Schur reduction: {error}"),
            })?
            .and_then(|rt| {
                let tiles = gam_gpu::pool::balanced_partition(rt, n);
                // Engage the device stacked-GEMM reduction when a MULTI-GPU pool can
                // overlap tiles, OR — the single-GPU gap this closes — when the one
                // stacked `(total_d×k)ᵀ(total_d×k)` GEMM clears the runtime's own
                // `gemm_min_flops`, so `try_fast_atb_on_ordinal` will actually offload
                // it instead of declining back to CPU. This reduction is the dense
                // build's O(n·d·k²) cost (measured on an H100 as ~28% of the fit in
                // `block_gemm_subtract`), and on a single GPU it previously always ran
                // on the CPU because the tile path required `len() > 1` — the device
                // sat idle. `assembly_work` IS the stacked GEMM's flop count (2·k²·Σd),
                // so this is exactly `try_fast_atb`'s own offload predicate; below the
                // GEMM floor the launch/staging tax loses to the CPU, so we keep the
                // deterministic CPU rayon fold there. Small K (e.g. K=8) never clears
                // the floor and stays on the CPU — magic-by-default crossover, no flag.
                let engage = tiles.len() > 1 || assembly_work >= rt.policy().gemm_min_flops as u128;
                (engage && !tiles.is_empty()).then_some(tiles)
            })
    };

    let Some(tiles) = tiles else {
        // Single-device / CPU. The per-row contributions `-Σ_i leftᵀ·right` fold
        // into the `k×k` `schur` independently — the same dense-assembly axis the
        // multi-GPU tile path partitions, and the dense-Direct analog of the
        // per-row matvec / streaming `accumulate_chunk` loops already parallelized
        // for #1017. A row's contribution touches only the columns where its two
        // factor rows are nonzero (`block_gemm_subtract` skips the rest, #1995), so
        // its cost is the factor materialization and block solve, not `d·k²`.
        //
        // Fan it across rayon over fixed row chunks: each chunk reduces its rows
        // (in row order) into a private zero-seeded `k×k` partial, then the
        // partials are folded into `schur` in CHUNK order. The per-chunk row order
        // and the inter-chunk fold order are both fixed independent of thread
        // scheduling, so the f64 reduction is **bit-identical run-to-run** (the
        // #1017 determinism gate). NOTE: bit-identical run-to-run does NOT make
        // it bit-identical to the in-place serial loop — the chunk-boundary
        // reassociation of the reduction sum is a genuine f64 departure (the
        // established equivalence `accumulate_chunk` / the per-row matvec operate
        // under, well inside the Newton solve's tolerance). It bounds candidate-
        // to-candidate drift to that reassociation margin, so the criterion
        // ranking is stable EXCEPT for candidates tying within the margin, where
        // the winner can flip; it is not an exact no-move guarantee (#1211). For
        // an exact-order guarantee, take the serial path. Stay in-place serial
        // below the row floor and when already inside a rayon worker (the topology
        // race fans candidates with `run_topology_race_parallel`) to avoid
        // nested-rayon oversubscription — the same guard the matvec uses.
        let n_rows = sys.rows.len();
        let parallel =
            n_rows >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            // Deterministic ordered fold: chunk partials hold `-Σ contribution`
            // over their rows, so `schur += partial` reproduces the serial
            // `schur -= Σ contribution` in fixed chunk order. Each partial folds
            // only the `(a, b)` pairs its rows touch, so the fold costs the touched
            // pairs rather than `k²` per chunk: with dense `k×k` partials the
            // main-thread fold was 66% of the build at `k = 4096`, 30,000 rows and
            // 28 active atoms (#2900, job 1148662). The governor admits as many
            // partials as its ledger takes, up to one per pool thread; below one,
            // the rows reduce in place. A declined footprint costs parallelism,
            // never per-product work: a keyed store that hashed every product ran
            // this solve at 24.5 s against 1.2 s dense (sw4i 1255679).
            let governor = gam_runtime::resource::MemoryGovernor::global();
            let (fold, charge) = plan_touched_pair_fold(
                k,
                fold_row_chunk_partial_count(n_rows),
                || governor.remaining_bytes(),
                |bytes| match governor.try_reserve(bytes, "reduced-Schur touched-pair partials") {
                    Ok(reservation) => Some(reservation),
                    Err(refusal) => {
                        log::trace!("[reduced Schur] {refusal}");
                        None
                    }
                },
            );
            if let TouchedPairFold::Chunked { partials } = fold {
                let folded = fold_touched_pair_chunk_partials(
                    sys,
                    htt_factors,
                    backend,
                    kind,
                    schur,
                    partials,
                );
                drop(charge);
                return folded;
            }
        }
        // Serial in-place reduction (original order) — bit-for-bit reference.
        for (i, row) in sys.rows.iter().enumerate() {
            subtract_row_schur_contribution(
                sys,
                i,
                row,
                htt_factors.factor(i),
                backend,
                kind,
                schur,
            )?;
        }
        return Ok(());
    };

    // Multi-GPU: one private `-Σ leftᵀ·right` partial per contiguous device
    // tile. Each tile runs on its own scoped worker thread that binds its
    // ordinal's context and issues a single stacked AᵀB GEMM on that device, so
    // the tiles' GEMMs overlap across the pool. Folding the partials back into
    // the H_ββ-seeded `schur` reproduces the serial reduction (up to inter-tile
    // reassociation).
    let partials: Result<Vec<Array2<f64>>, ArrowSchurError> = std::thread::scope(|scope| {
        let handles: Vec<_> = tiles
            .iter()
            .map(|(ordinal, range)| {
                let ordinal = *ordinal;
                let range = range.clone();
                scope.spawn(move || {
                    // Bind this ordinal's CUDA context on this worker thread so
                    // the per-row GPU GEMM shims issued from `tile_schur_partial`
                    // offload to that device. A missing context or bind failure
                    // is intentionally consumed without escalation — the shims
                    // no-op back to CPU and the math is unchanged. Off Linux
                    // runtime resolution is always absent, so this branch
                    // is unreachable and the bind is omitted entirely.
                    #[cfg(target_os = "linux")]
                    {
                        if let Some(ctx) = gam_gpu::device_runtime::cuda_context_for(ordinal) {
                            if let Err(err) = ctx.bind_to_thread() {
                                log::trace!(
                                    "arrow-schur tile {ordinal}: CUDA context bind failed ({err}); \
                                     this tile reduces on the CPU"
                                );
                            }
                        }
                    }
                    tile_schur_partial(sys, htt_factors, backend, kind, ordinal, range)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| ArrowSchurError::SchurFactorFailed {
                        reason: "schur-reduction tile thread panicked".to_string(),
                    })?
            })
            .collect()
    });
    let partials = partials?;

    // Fold partials into `schur` in tile order (contiguous, covering 0..n) so
    // the per-tile and inter-tile accumulation order is the row order; each
    // partial holds `-Σ contribution` over its rows, so `schur += partial`
    // reproduces `schur -= Σ contribution`.
    for partial in &partials {
        for a in 0..k {
            for b in 0..k {
                schur[[a, b]] += partial[[a, b]];
            }
        }
    }
    Ok(())
}

/// #2731/#2900 — a dense `k × k` route for a reduced-Schur `log|S|`, priced against the
/// matrix-free route that is its alternative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseReducedSchurRoute {
    /// The criterion lane materializes `S` with `k` operator products, factors it, and
    /// inverts the factor for its derivative bundle.
    Lane,
    /// The chunked evidence route pulls every row's elimination term back into `S`, at
    /// `pullback_flops`, then factors it.
    ChunkedEvidence { pullback_flops: u64 },
}

/// #2731/#2900 — how a reduced-Schur `log|S|` of dimension `k` may use its matrix-free
/// route before the dense `route`, when one matrix-free product of the reduced Schur costs
/// `apply_flops`. This is gam-linalg's dense-route work model: `PcgAttempt::Only` when the
/// route's blocks exceed the memory governor's single-materialization cap, otherwise
/// `PcgAttempt::Budgeted { products }`, the dense route's `build + k³/3` flops in products.
///
/// Either route holds about six `k × k` blocks at its peak. The lane holds the applied
/// operator, the factorization's working copy, its inverse or eigenvectors, the derivative
/// bundle, and the EFS probes with their inverse images. The chunked route holds three
/// accumulators, each chunk's Schur and two classification metrics, and the log-det's two
/// metric clones and factor. The lane's build is its `k` products plus the triangular
/// inverse's `k³/3`; the chunked route's is its pullback. A caller whose alternative has a
/// fixed product count, such as SLQ, takes the dense route when `products` is below it; a
/// caller whose alternative is iterative spends at most `products` before the dense route.
pub fn dense_reduced_schur_route(
    route: DenseReducedSchurRoute,
    k: usize,
    apply_flops: u64,
) -> gam_linalg::pcg::PcgAttempt {
    dense_reduced_schur_route_under_cap(
        route,
        k,
        apply_flops,
        gam_runtime::resource::MemoryGovernor::global().single_materialization_cap_bytes(),
    )
}

/// [`dense_reduced_schur_route`] against an explicit materialization cap.
pub(crate) fn dense_reduced_schur_route_under_cap(
    route: DenseReducedSchurRoute,
    k: usize,
    apply_flops: u64,
    cap_bytes: usize,
) -> gam_linalg::pcg::PcgAttempt {
    const DENSE_ROUTE_BLOCKS: usize = 6;
    let dim = k as u64;
    let build = match route {
        DenseReducedSchurRoute::Lane => dim
            .saturating_mul(apply_flops)
            .saturating_add(dim.saturating_mul(dim).saturating_mul(dim) / 3),
        DenseReducedSchurRoute::ChunkedEvidence { pullback_flops } => pullback_flops,
    };
    gam_linalg::pcg::DenseRouteWork {
        build,
        apply: apply_flops,
    }
    .pcg_attempt_under_cap(k, cap_bytes / DENSE_ROUTE_BLOCKS)
}

/// #2900 row 6.16 — whether a surrogate lane takes the dense `k × k` reduced Schur rather
/// than its frozen rational surrogate, when one reduced-Schur product costs
/// `reduced_schur_apply_flops`. The rational surrogate spends at most `num_probes · k`
/// products per evaluation, the budget the device operator is sized against. The dense
/// lane is priced as [`DenseReducedSchurRoute::Lane`]: it is taken only where that build,
/// in products, is within the surrogate's budget and its blocks fit the memory governor's
/// single-materialization cap. Before this, the dense lane was taken wherever its blocks
/// fit in core, whatever it cost.
pub fn surrogate_lane_prices_dense_reduced_schur(
    lane: &SurrogateLaneState,
    k: usize,
    reduced_schur_apply_flops: u64,
) -> bool {
    let rational_products = lane.cfg.num_probes.saturating_mul(k);
    matches!(
        dense_reduced_schur_route(DenseReducedSchurRoute::Lane, k, reduced_schur_apply_flops),
        gam_linalg::pcg::PcgAttempt::Budgeted { products } if products <= rational_products
    )
}

pub(crate) fn build_dense_schur_direct<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    gpu_policy: gam_gpu::GpuPolicy,
) -> Result<Array2<f64>, ArrowSchurError> {
    build_dense_schur_direct_under_cap(
        sys,
        htt_factors,
        ridge_beta,
        backend,
        gpu_policy,
        gam_runtime::resource::MemoryGovernor::global().single_materialization_cap_bytes(),
    )
}

/// [`build_dense_schur_direct`] against an explicit materialization cap.
pub(crate) fn build_dense_schur_direct_under_cap<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    gpu_policy: gam_gpu::GpuPolicy,
    cap_bytes: usize,
) -> Result<Array2<f64>, ArrowSchurError> {
    let k = sys.k;
    // Materialise H_ββ via the BetaPenaltyOp trait (#296): DensePenaltyOp
    // for the legacy dense path, structured ops for SAE / Kronecker smooths.
    let op = sys.effective_penalty_op();
    if op.dim() != k {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "Direct BA requires a K×K shared H_ββ penalty operator".to_string(),
        });
    }
    // Fail LOUD, never OOM-kill (#1017): the dense reduced Schur is `k × k` f64.
    // At SAE LLM borders (qwen `k = 98304` ⇒ 77 GiB) materialising it would crash
    // the host. Direct deliberately uses this one canonical dense Schur for both
    // the Newton step and evidence; large-border matrix-free solves belong to
    // InexactPCG. Refuse an explicit oversized Direct request with an actionable
    // error rather than duplicating ownership or degrading silently into an OOM.
    // The ceiling is the memory governor's single-materialization cap, the one
    // routing threshold a dense materialization in this process reads.
    let dense_bytes = (k as u128).saturating_mul(k as u128).saturating_mul(8);
    if dense_bytes > cap_bytes as u128 {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "dense reduced Schur is {k}×{k} f64 = {} MiB, exceeding the memory governor's \
                 {} MiB single-materialization cap; Direct requires one canonical dense Schur \
                 for its step and evidence; select InexactPCG for a matrix-free large-border step",
                dense_bytes / (1024 * 1024),
                cap_bytes / (1024 * 1024),
            ),
        });
    }
    let mut schur = op.to_dense();
    for j in 0..k {
        schur[[j, j]] += ridge_beta;
    }
    reduce_row_schur_contributions(
        sys,
        htt_factors,
        backend,
        SchurReductionKind::Direct,
        &mut schur,
        gpu_policy,
    )?;
    symmetrize_upper_from_lower(&mut schur);
    Ok(schur)
}

pub(crate) fn build_dense_schur_sqrt_ba<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    gpu_policy: gam_gpu::GpuPolicy,
) -> Result<Array2<f64>, ArrowSchurError> {
    build_dense_schur_sqrt_ba_under_cap(
        sys,
        htt_factors,
        ridge_beta,
        backend,
        gpu_policy,
        gam_runtime::resource::MemoryGovernor::global().single_materialization_cap_bytes(),
    )
}

/// [`build_dense_schur_sqrt_ba`] against an explicit materialization cap.
pub(crate) fn build_dense_schur_sqrt_ba_under_cap<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    gpu_policy: gam_gpu::GpuPolicy,
    cap_bytes: usize,
) -> Result<Array2<f64>, ArrowSchurError> {
    let k = sys.k;
    // Materialise H_ββ via the BetaPenaltyOp trait (#296).
    let op = sys.effective_penalty_op();
    if op.dim() != k {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "Square-Root BA direct solve requires a K×K shared H_ββ penalty operator"
                .to_string(),
        });
    }
    // Same fail-loud host-memory contract as the Direct reduction (#1017).  The
    // square-root BA route still materialises the same dense `k×k` reduced
    // Schur; letting this path bypass the cap would preserve an OOM-class
    // fallback even after Direct learned to refuse matrix-free-only borders.
    let dense_bytes = (k as u128).saturating_mul(k as u128).saturating_mul(8);
    if dense_bytes > cap_bytes as u128 {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "square-root BA dense reduced Schur is {k}×{k} f64 = {} MiB, exceeding the \
                 memory governor's {} MiB single-materialization cap; this border is \
                 matrix-free-only",
                dense_bytes / (1024 * 1024),
                cap_bytes / (1024 * 1024),
            ),
        });
    }
    let mut schur = op.to_dense();
    for j in 0..k {
        schur[[j, j]] += ridge_beta;
    }
    reduce_row_schur_contributions(
        sys,
        htt_factors,
        backend,
        SchurReductionKind::SqrtBa,
        &mut schur,
        gpu_policy,
    )?;
    symmetrize_upper_from_lower(&mut schur);
    Ok(schur)
}

/// Spectral positive-definiteness floor for the reduced Schur complement
/// `S` (#1026 SAE co-collapse SOLVE-path cure).
///
/// Reached only after the genuine Cholesky of `S` has REFUSED it (an indefinite
/// reduced Schur: collapsed atoms drive a per-row `H_tt` near-singular, so the
/// accumulated `Σ_i H_tβᵀ (H_tt)⁻¹ H_tβ` over-subtracts `H_ββ + ridge_β·I` into a
/// matrix with a non-positive eigenvalue). Rather than reject and let the LM
/// loop inflate `ridge_β` over EVERY β direction (the #1026 "crawl"), we
/// symmetric-eigendecompose `S` and clamp every eigenvalue UP to
/// `floor·max(λ)`. This is Levenberg–Marquardt restricted to exactly the
/// indefinite/collapsed subspace: a well-separated positive direction
/// (`λ ≫ floor·max λ`) keeps its EXACT eigenvalue (`λ.max(floor·max λ) = λ`), so
/// the Newton step in the healthy β subspace is unchanged, while only the
/// collapsed directions get the minimal positive stiffness needed for a PD
/// solve. Returns the floored, symmetric, strictly-PD matrix, or `None` if `S`
/// has no usable scale (non-finite / all-zero spectrum), in which case the
/// caller keeps the strict refusal.
///
/// Mirrors the per-row evidence floor
/// [`super::factorization::factor_spectral_deflated_criterion_row_with_geometry`]; the only
/// difference is the floored VALUE — a small positive `floor·max λ` (Tikhonov,
/// for an accurate solve) here, vs unit stiffness `+1` (`log 1 = 0`) there (for
/// the quotient log-det).
pub(crate) fn spectral_pd_floored_schur(
    schur: &Array2<f64>,
    relative_floor: f64,
) -> Option<(Array2<f64>, Array2<f64>)> {
    spectral_pd_floored_schur_with_factor(schur, relative_floor)
}

/// Shared body for [`spectral_pd_floored_schur`]: symmetrise, eigendecompose,
/// condition the spectrum, and return BOTH
/// the conditioned matrix `Σ λ̃_i v_i v_iᵀ` (consumed by Steihaug / matvec)
/// and its lower Cholesky factor.
///
/// The factor is built DIRECTLY from the conditioned spectral form — QR of
/// `W = diag(√λ̃)·Vᵀ` gives `A = WᵀW = RᵀR`, so `L = Rᵀ` — never by
/// re-factorising the reconstructed matrix. Reconstruct-then-refactor fails
/// under extreme eigenvalue spread: with `λ_max ~ 1e57` the `Σ λ̃ v vᵀ`
/// reconstruction carries `O(ε·λ_max)` round-off, which swamps unit-deflated
/// (`λ̃ = 1`) and floored (`λ̃ = floor·λ_max`) directions and re-poisons the
/// second Cholesky — the #2230 "spectral PD-floor reconstruction still non-PD"
/// refusal at a ρ whose conditioned evidence is perfectly well-defined. The QR
/// route factors the exact conditioned spectrum, so it succeeds whenever the
/// policy produced strictly positive `λ̃` (always, by construction).
fn spectral_pd_floored_schur_with_factor(
    schur: &Array2<f64>,
    relative_floor: f64,
) -> Option<(Array2<f64>, Array2<f64>)> {
    let n = schur.nrows();
    if n == 0 || schur.ncols() != n || !(relative_floor.is_finite() && relative_floor > 0.0) {
        return None;
    }
    // Symmetrise defensively (the assembled Schur is symmetric up to reduction
    // order; the eig routine assumes exact symmetry).
    let mut sym = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let v = 0.5 * (schur[[i, j]] + schur[[j, i]]);
            if !v.is_finite() {
                return None;
            }
            sym[[i, j]] = v;
        }
    }
    let (evals, evecs) = sym.eigh(Side::Lower).ok()?;
    let max_abs = evals.iter().fold(
        0.0_f64,
        |acc, &v| if v.is_finite() { acc.max(v.abs()) } else { acc },
    );
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return None;
    }
    let floor = relative_floor * max_abs;
    // Newton-step policy (LM): clamp every eigenvalue UP to a strictly positive
    // `floor` — healthy positive directions (`λ ≫ floor`) keep their EXACT
    // eigenvalue, collapsed/indefinite directions get the minimal stiffness for
    // a stable `Δβ`.
    let mut conditioned = Array2::<f64>::zeros((n, n));
    let mut weighted_vt = Array2::<f64>::zeros((n, n));
    for eig_idx in 0..evals.len() {
        let lambda = evals[eig_idx];
        let lambda_conditioned = if lambda.is_finite() {
            lambda.max(floor)
        } else {
            floor
        };
        let sqrt_lambda = lambda_conditioned.sqrt();
        for i in 0..n {
            let vi = evecs[[i, eig_idx]];
            weighted_vt[[eig_idx, i]] = sqrt_lambda * vi;
            if vi == 0.0 {
                continue;
            }
            for j in 0..n {
                conditioned[[i, j]] += lambda_conditioned * vi * evecs[[j, eig_idx]];
            }
        }
    }
    let factor =
        spectral_qr_cholesky_factor(&weighted_vt).or_else(|| cholesky_lower(&conditioned).ok())?;
    Some((conditioned, factor))
}

/// Original-coordinate unit-deflation for an evidence reduced Schur.
///
/// The rank decision and unit pin are made in the caller's β coordinates. A
/// Jacobi congruence is appropriate for a Newton solve but would turn a unit
/// eigenvalue in scaled coordinates into a scale-dependent stiffness after
/// unscaling, corrupting both `log 1 = 0` and the cached null-space metadata.
fn factor_evidence_unit_deflated_schur(
    schur: &Array2<f64>,
    relative_floor: f64,
    refuse_resolved_indefinite: bool,
    exact_a: Option<&ExactAReducedClassification>,
) -> Result<DenseReducedSchurFactorization, ArrowSchurError> {
    let declined = |reason: &str| ArrowSchurError::SchurFactorFailed {
        reason: format!("evidence reduced Schur unit-deflation declined ({reason})"),
    };
    let n = schur.nrows();
    if n == 0 || schur.ncols() != n || !(relative_floor.is_finite() && relative_floor > 0.0) {
        return Err(declined("empty, non-square, or invalid relative floor"));
    }
    let mut sym = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let value = 0.5 * (schur[[i, j]] + schur[[j, i]]);
            if !value.is_finite() {
                return Err(declined("non-finite entry"));
            }
            sym[[i, j]] = value;
        }
    }
    let (raw_evals, evecs) = sym
        .eigh(Side::Lower)
        .map_err(|_| declined("symmetric eigendecomposition failed"))?;
    let max_abs = raw_evals.iter().fold(0.0_f64, |acc, &value| {
        if value.is_finite() {
            acc.max(value.abs())
        } else {
            acc
        }
    });
    if !(max_abs.is_finite() && max_abs > 0.0) {
        return Err(declined("no usable spectrum"));
    }
    if let Some(geometry) = exact_a
        && (geometry.majorizer_metric.dim() != (n, n)
            || geometry.clamp_metric.dim() != (n, n))
    {
        return Err(declined(
            "exact-A majorizer/clamp metrics do not match the reduced Schur",
        ));
    }
    if refuse_resolved_indefinite && exact_a.is_none() {
        return Err(declined(
            "exact-A evidence classification requires its raw B/delta/clamp carrier",
        ));
    }
    let deflate_floor = relative_floor * max_abs * (1.0 - SPECTRAL_DEFLATION_HYSTERESIS_FRACTION);
    let mut conditioning = vec![BetaSchurSpectralConditioning::Raw; raw_evals.len()];
    let mut cond_evals = raw_evals.clone();
    let mut classification_changed = false;
    for eig_idx in 0..raw_evals.len() {
        let value = raw_evals[eig_idx];
        if let Some(geometry) = exact_a {
            let direction = evecs.column(eig_idx);
            let majorizer_curvature =
                direction.dot(&geometry.majorizer_metric.dot(&direction));
            let clamp_curvature = direction.dot(&geometry.clamp_metric.dot(&direction));
            match classify_exact_a_direction(
                value,
                n,
                max_abs,
                majorizer_curvature,
                clamp_curvature,
            ) {
                ExactADirectionClassification::ResolvedPositive { curvature } => {
                    cond_evals[eig_idx] = curvature;
                }
                ExactADirectionClassification::NumericalNull => {
                    conditioning[eig_idx] = BetaSchurSpectralConditioning::UnitDeflated;
                    cond_evals[eig_idx] = 1.0;
                    classification_changed = true;
                }
                ExactADirectionClassification::ClampBasin { curvature } => {
                    conditioning[eig_idx] = BetaSchurSpectralConditioning::ClampBasin;
                    cond_evals[eig_idx] = curvature;
                    classification_changed = true;
                }
                ExactADirectionClassification::Saddle { curvature, basin } => {
                    return Err(ArrowSchurError::SchurFactorFailed {
                        reason: format!(
                            "reduced-Schur {}: direction {eig_idx} has raw exact-A curvature \
                             {curvature:.6e} and clamp basin {basin:.6e}; the shared \
                             majorizer-metric classifier declares a genuine saddle (#2515/#2336)",
                            ArrowSchurError::indefinite_evidence_marker(),
                        ),
                    });
                }
            }
        } else if !value.is_finite() || value < deflate_floor {
            conditioning[eig_idx] = BetaSchurSpectralConditioning::UnitDeflated;
            cond_evals[eig_idx] = 1.0;
            classification_changed = true;
        }
    }

    // Preserve the ordinary equilibrated-Cholesky bit path in the interior.
    // If Cholesky alone is numerically unable to factor a spectrally healthy
    // operator, the spectral QR below still factors the identical raw spectrum.
    if !classification_changed
        && let Ok(interior) = factor_dense_reduced_schur(schur, ReducedSchurPolicy::StrictNewton)
    {
        return Ok(interior);
    }

    let mut conditioned = Array2::<f64>::zeros((n, n));
    let mut weighted_vt = Array2::<f64>::zeros((n, n));
    for eig_idx in 0..n {
        let lambda = cond_evals[eig_idx];
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(declined("conditioned eigenvalue is not finite and positive"));
        }
        let sqrt_lambda = lambda.sqrt();
        for i in 0..n {
            let vi = evecs[[i, eig_idx]];
            weighted_vt[[eig_idx, i]] = sqrt_lambda * vi;
            if vi != 0.0 {
                for j in 0..n {
                    conditioned[[i, j]] += lambda * vi * evecs[[j, eig_idx]];
                }
            }
        }
    }
    let factor = spectral_qr_cholesky_factor(&weighted_vt)
        .ok_or_else(|| declined("spectral QR Cholesky of the conditioned spectrum declined"))?;
    let beta_conditioning = classification_changed.then(|| BetaSchurConditioningSpectrum {
        evecs,
        raw_evals,
        cond_evals,
        conditioning: conditioning.into(),
    });
    Ok(DenseReducedSchurFactorization {
        factor,
        conditioned_schur: beta_conditioning.as_ref().map(|_| conditioned),
        beta_conditioning,
    })
}

/// Lower Cholesky factor of `A = WᵀW` computed from `W` itself: QR gives
/// `W = QR ⇒ A = RᵀR`, so the factor is `L = Rᵀ` (rows sign-fixed to a positive
/// diagonal). `W` here is `diag(√λ̃)·Vᵀ` with every `λ̃ > 0`, so `W` has full
/// rank and the factor exists exactly; returns `None` only if the QR itself
/// declines or produces a non-finite / zero pivot, in which case the caller
/// falls back to factoring the reconstructed matrix (the historical path).
fn spectral_qr_cholesky_factor(weighted_vt: &Array2<f64>) -> Option<Array2<f64>> {
    let n = weighted_vt.nrows();
    let (_q, r) = weighted_vt.qr().ok()?;
    if r.nrows() != n || r.ncols() != n {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let d = r[[i, i]];
        if !d.is_finite() || d == 0.0 {
            return None;
        }
        let s = if d < 0.0 { -1.0 } else { 1.0 };
        for j in i..n {
            let v = s * r[[i, j]];
            if !v.is_finite() {
                return None;
            }
            l[[j, i]] = v;
        }
    }
    Some(l)
}

/// Jacobi/Van der Sluis diagonal equilibration scale for a symmetric matrix
/// (#2015): `d_a = sqrt(|schur[a,a]|)`. An exactly zero or non-finite diagonal
/// carries no scale, so that coordinate is left unscaled (`d_a = 1`) rather
/// than divided by a picked substitute (#2469). This is a PURE
/// numerical-conditioning aid for [`factor_dense_reduced_schur`] below — it is
/// never returned or exposed, and it changes no value any caller of that
/// function sees, only the accuracy of computing it.
///
/// #2822 — the scale is the diagonal's MAGNITUDE, not the signed entry. Van der
/// Sluis is stated for a positive-definite matrix, where the two agree
/// (`|S_aa| = S_aa`, and `abs` on a positive finite double is exact), so this is
/// BIT-IDENTICAL on every matrix that reaches the Cholesky success path — a
/// positive-definite matrix has no non-positive diagonal. It differs only on the
/// matrices that fall through to the spectral floor, and there it is the whole
/// point.
///
/// Reading the SIGNED entry made the equilibration ANTI-equilibrating on exactly
/// the operators the floor exists for. A collapsed reduced Schur carries a
/// NEGATIVE diagonal; the signed test `S_aa > 1e-18` then in place was false, so that
/// direction was scaled by the substitute `√1e-18 = 1e-9` — dividing an entry of
/// magnitude `|S_aa|` by `1e-18` and AMPLIFYING it by eighteen decades instead of
/// normalising it to unit magnitude. `spectral_pd_floored_schur` then reads
/// `floor = relative_floor · max|λ|` off that inflated spectrum, so the floor is
/// eighteen decades too high and clamps the HEALTHY directions with it.
///
/// Measured on the `owed_1026` mixed-collapse fixture `S = diag(+5, −99)`, whose
/// healthy Newton step is exactly `Δβ_0 = −g/S = 10/5 = 2`: the signed form gave
/// `d = (√5, 1e-9)`, `S̃ = diag(1, −9.9e19)`, `floor = 1e-8 · 9.9e19 = 9.9e11`,
/// so the healthy `λ̃ = 1` was clamped to `9.9e11`, `S_floored,00 = 9.9e11·5 =
/// 4.95e12` and `Δβ_0 = 2.0202020202e-12` — the live subspace wrong by twelve
/// orders of magnitude, against a documented contract that it keeps its EXACT
/// eigenvalue. With the magnitude, `S̃ = diag(1, −1)`, `floor = 1e-8`, the healthy
/// direction keeps `λ̃ = 1` and `Δβ_0 = 2` exactly, while the collapsed direction
/// still receives its minimal positive stiffness.
fn jacobi_diagonal_scale(schur: &Array2<f64>) -> Array1<f64> {
    let n = schur.nrows();
    let mut d = Array1::<f64>::zeros(n);
    for a in 0..n {
        let magnitude = schur[[a, a]].abs();
        d[a] = if magnitude.is_finite() && magnitude > 0.0 {
            magnitude.sqrt()
        } else {
            1.0
        };
    }
    d
}

/// Factor the dense reduced Schur complement `S`, returning its lower Cholesky
/// factor, the conditioned operator when policy changed it, and authoritative
/// β-null metadata for evidence unit deflation.
///
/// #2015 — SOLVER-LEVEL conditioning fix (design: issue 2015 comment
/// 4949898801). A real activation+behavior augmented target can carry output
/// column-norm spreads of ~1e4 (joint Hessian condition number ≈ 1e8), which a
/// PLAIN `cholesky_lower(schur)` is not designed to survive: the recursive
/// `L_ii = sqrt(S_ii − Σ_{j<i} L_ij²)` step loses precision (or falsely
/// refuses a genuinely PD matrix) when the diagonal spans many orders of
/// magnitude. Equilibrate FIRST: `D = diag(d)` with `d_a = sqrt(|S_aa|)`
/// ([`jacobi_diagonal_scale`] — Van der Sluis equilibration, provably within a
/// factor of `n` of the OPTIMAL diagonal preconditioner for a symmetric
/// matrix), factor `S̃ = D⁻¹SD⁻¹` (unit diagonal by construction) with the
/// EXACT SAME Cholesky/spectral-floor logic below, then undo the equilibration
/// on the way out.
///
/// This is NOT a reparametrization of any objective or estimand (contrast the
/// REVERTED #2015 attempt that divided the FIT TARGET's columns, which
/// changed what "best fit" means for a homoscedastic residual). `D` is
/// diagonal, so `L := D·L̃` is STILL lower-triangular, and
/// `L·Lᵀ = D·S̃·Dᵀ = D·(D⁻¹SD⁻¹)·D = S` exactly — `L` is a bit-exact valid
/// Cholesky factor of the CALLER'S ORIGINAL `schur`, just computed via a
/// numerically superior route. Undoing the scale is one exact elementwise
/// multiply (`factor[i,j] *= d[i]`, `floored[i,j] *= d[i]*d[j]`) — no further
/// precision is lost recovering original units. Evidence unit deflation
/// deliberately bypasses this congruence and works in the original β
/// coordinates so a unit-pinned null contributes exactly `log 1`.
///
/// GPU cross-reference: the device/GPU dense-reference path
/// (`gam_solve::gpu_kernels::arrow_schur::solve_arrow_newton_step_dense_reference`)
/// factors the full joint `(t, β)` system independently of this function and
/// does NOT yet get this equilibration. Both paths are exact; the GPU path is
/// simply not yet as well-conditioned on an ill-scaled system. Porting the
/// same technique there is a deliberate follow-up, not part of this change.
///
/// Newton-step damping and evidence quotient deflation are deliberately
/// different policies: Tikhonov directions retain a small positive curvature
/// for a stable step, while evidence-null directions are pinned to unit
/// stiffness so their log-determinant contribution is exactly zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ReducedSchurPolicy {
    StrictNewton,
    NewtonTikhonov { relative_floor: f64 },
    EvidenceUnitDeflation {
        relative_floor: f64,
        /// #2515 — refuse a RESOLVED negative direction instead of unit-pinning
        /// it. See [`ArrowEvidencePolicy::UnitDeflationRefusingIndefinite`].
        refuse_resolved_indefinite: bool,
    },
}

impl ReducedSchurPolicy {
    pub(crate) fn newton(relative_floor: Option<f64>) -> Self {
        match relative_floor {
            Some(relative_floor) => Self::NewtonTikhonov { relative_floor },
            None => Self::StrictNewton,
        }
    }
}

#[derive(Debug)]
pub(crate) struct DenseReducedSchurFactorization {
    pub(crate) factor: Array2<f64>,
    pub(crate) conditioned_schur: Option<Array2<f64>>,
    pub(crate) beta_conditioning: Option<BetaSchurConditioningSpectrum>,
}

/// Majorizer and clamp quadratic forms on the exact-A Schur graph
/// `t(beta) = -A_tt^{-1} A_tbeta beta`.  They let the reduced route feed the
/// same scalar direction classifier as the dense joint route instead of
/// substituting a local relative eigenvalue test (#2515).
pub(crate) struct ExactAReducedClassification {
    pub(crate) majorizer_metric: Array2<f64>,
    pub(crate) clamp_metric: Array2<f64>,
}

/// The installed exact-A carrier, checked against the system's row count.
fn exact_a_geometry(
    sys: &ArrowSchurSystem,
) -> Result<Option<&ExactAClassificationGeometry>, ArrowSchurError> {
    let Some(geometry) = sys.exact_a_classification.as_ref() else {
        return Ok(None);
    };
    if geometry.rows.len() != sys.rows.len() {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "exact-A classification carries {} rows for an {}-row system",
                geometry.rows.len(),
                sys.rows.len(),
            ),
        });
    }
    Ok(Some(geometry))
}

/// #2828 — `delta_beta = A_ββ − B_ββ` materialized on the border, when installed.
fn exact_a_border_remainder(
    geometry: &ExactAClassificationGeometry,
    k: usize,
) -> Result<Option<Array2<f64>>, ArrowSchurError> {
    let Some(remainder) = geometry.border_remainder.as_ref() else {
        return Ok(None);
    };
    if remainder.dim() != k {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "exact-A classification border remainder has width {} for border width {k}",
                remainder.dim(),
            ),
        });
    }
    Ok(Some(remainder.to_dense()))
}

/// One row's operands on the exact-A Schur graph: `graph = −A_tt⁻¹A_tβ` (`q × k`), the
/// materialized cross block `A_tβ`, and the majorizer blocks `B_tβ = A_tβ − ΔC_tβ` and
/// `B_tt = A_tt − ΔC_tt`.
struct ExactARowLift {
    graph: Array2<f64>,
    a_tbeta: Array2<f64>,
    b_tbeta: Array2<f64>,
    b_tt: Array2<f64>,
}

fn exact_a_row_lift(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    geometry: &ExactAClassificationGeometry,
    row_idx: usize,
) -> Result<ExactARowLift, ArrowSchurError> {
    let k = sys.k;
    let row = &sys.rows[row_idx];
    let q = sys.row_dims[row_idx];
    let operands = &geometry.rows[row_idx];
    if operands.delta_tt.dim() != (q, q)
        || operands.delta_tbeta.nrows() != q
        || operands.delta_tbeta.ncols() != operands.border_columns.len()
        || operands.clamp_diag.len() != q
    {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "exact-A classification row {row_idx} is incompatible with row width {q} and border width {k}",
            ),
        });
    }
    let a_tbeta = sys_htbeta_materialize_row(sys, row_idx, row)?;
    let mut b_tbeta = a_tbeta.clone();
    for (carrier_col, &system_col) in operands.border_columns.iter().enumerate() {
        if system_col >= k {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "exact-A classification border index {system_col} exceeds width {k}",
                ),
            });
        }
        for local in 0..q {
            b_tbeta[[local, system_col]] -= operands.delta_tbeta[[local, carrier_col]];
        }
    }
    let b_tt = &row.htt - &operands.delta_tt;
    let mut graph = Array2::<f64>::zeros((q, k));
    for col in 0..k {
        let solved = cholesky_solve_vector(htt_factors.factor(row_idx), a_tbeta.column(col));
        for local in 0..q {
            graph[[local, col]] = -solved[local];
        }
    }
    Ok(ExactARowLift {
        graph,
        a_tbeta,
        b_tbeta,
        b_tt,
    })
}

pub(crate) fn exact_a_reduced_classification(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
) -> Result<Option<ExactAReducedClassification>, ArrowSchurError> {
    let Some(geometry) = exact_a_geometry(sys)? else {
        return Ok(None);
    };
    let k = sys.k;
    let mut majorizer_metric = sys.effective_penalty_op().to_dense();
    let mut clamp_metric = Array2::<f64>::zeros((k, k));
    // #2828 — the shared block is `A_ββ = B_ββ + delta_beta`: the majorizer
    // metric is `B`'s, and `E_ββ = -delta_beta` is border clamp curvature.
    if let Some(remainder) = exact_a_border_remainder(geometry, k)? {
        majorizer_metric -= &remainder;
        clamp_metric -= &remainder;
    }
    for row_idx in 0..sys.rows.len() {
        let ExactARowLift {
            graph,
            b_tbeta,
            b_tt,
            ..
        } = exact_a_row_lift(sys, htt_factors, geometry, row_idx)?;
        majorizer_metric += &graph.t().dot(&b_tt.dot(&graph));
        majorizer_metric += &graph.t().dot(&b_tbeta);
        majorizer_metric += &b_tbeta.t().dot(&graph);
        let clamp_diag = &geometry.rows[row_idx].clamp_diag;
        let weighted_graph = Array2::from_shape_fn(graph.dim(), |(local, col)| {
            clamp_diag[local] * graph[[local, col]]
        });
        clamp_metric += &graph.t().dot(&weighted_graph);
    }
    Ok(Some(ExactAReducedClassification {
        majorizer_metric,
        clamp_metric,
    }))
}

/// The operands of the exact-`A` pencil on the reduced border (#2933 F07).
///
/// A route that eliminates the coordinate block classifies the Ritz pencil of `(A, Φ)` on
/// the `A`-lift `z(v) = (graph·v, v)`, `graph = −A_tt⁻¹A_tβ`, where `Φ` is the evidence
/// factor of the majorizer `B_raw`: its rows are what the evidence row factorization
/// returns for `B_tt`, gauge and spectral pins included, and its cross and border blocks
/// are `B`'s. The four forms are `k × k`:
///
/// * `metric`: `Y = zᵀΦz`;
/// * `substituted_metric`: `zᵀ(Φ − B_raw)z` over the rows' SPECTRAL pins, which raise the
///   positive band edge ([`exact_a_band_edge`]). Gauge pins and the border quotient enter
///   `Y` only. This route has no border spectral pin, so the border part is zero;
/// * `clamp_metric`: `zᵀEz`, the clamp curvature the majorizer omits;
/// * `lift_gram`: `I + Σ graphᵢᵀgraphᵢ`, so `‖z(v)‖² = vᵀ·lift_gram·v`.
///
/// With a border quotient `Q` installed the reduced operator is `P S_A P + QQᵀ`, so
/// `metric` and `lift_gram` are pinned the same way and the other two forms are projected:
/// a declared quotient direction reads `μ = 1` with no substituted stiffness.
/// `joint_dimension` and the Frobenius norms of the unpinned joint `A`, `Φ` and `E` are
/// the operands of [`exact_a_pencil_resolution`].
#[derive(Debug, Clone)]
pub struct ExactAReducedPencilOperands {
    pub metric: Array2<f64>,
    pub substituted_metric: Array2<f64>,
    pub clamp_metric: Array2<f64>,
    pub lift_gram: Array2<f64>,
    pub joint_dimension: usize,
    pub operator_frobenius: f64,
    pub metric_frobenius: f64,
    pub clamp_frobenius: f64,
}

/// [`ExactAReducedPencilOperands`] of an exact-`A` system, lifted through `htt_factors`,
/// the row factors its reduced Schur is eliminated with. A system without the raw
/// `B`/`ΔC`/clamp carrier is refused.
pub fn exact_a_reduced_pencil_operands(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
) -> Result<ExactAReducedPencilOperands, ArrowSchurError> {
    let Some(geometry) = exact_a_geometry(sys)? else {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "the exact-A reduced pencil requires the raw B/delta/clamp classification \
                     carrier"
                .to_string(),
        });
    };
    let k = sys.k;
    let frobenius_sq = |block: &Array2<f64>| block.iter().map(|value| value * value).sum::<f64>();
    let exact_border = sys.effective_penalty_op().to_dense();
    let mut operator_frobenius_sq = frobenius_sq(&exact_border);
    let mut metric = exact_border;
    let mut clamp_metric = Array2::<f64>::zeros((k, k));
    let mut clamp_frobenius_sq = 0.0_f64;
    if let Some(remainder) = exact_a_border_remainder(geometry, k)? {
        metric -= &remainder;
        clamp_metric -= &remainder;
        clamp_frobenius_sq += frobenius_sq(&remainder);
    }
    let mut metric_frobenius_sq = frobenius_sq(&metric);
    let mut substituted_metric = Array2::<f64>::zeros((k, k));
    let mut lift_gram = Array2::<f64>::eye(k);
    let mut joint_dimension = k;
    for row_idx in 0..sys.rows.len() {
        let q = sys.row_dims[row_idx];
        let ExactARowLift {
            graph,
            a_tbeta,
            b_tbeta,
            b_tt,
        } = exact_a_row_lift(sys, htt_factors, geometry, row_idx)?;
        joint_dimension += q;
        operator_frobenius_sq +=
            frobenius_sq(&sys.rows[row_idx].htt) + 2.0 * frobenius_sq(&a_tbeta);
        // The arguments `factor_blocks_for_system` factors a majorizer row with on the
        // evidence path, so `Φ_tt = LLᵀ` carries exactly the pins the evidence factor does.
        let pinned = factor_one_row_result(
            &ArrowRowBlock {
                htt: b_tt,
                htbeta: Array2::<f64>::zeros((q, 0)),
                gt: Array1::<f64>::zeros(q),
            },
            0.0,
            q,
            row_idx,
            true,
            sys.row_gauge_deflation
                .as_ref()
                .map_or(&[], |deflation| deflation.row(row_idx)),
            true,
            false,
            None,
        )?;
        let metric_tt = pinned.factor.dot(&pinned.factor.t());
        metric_frobenius_sq += frobenius_sq(&metric_tt) + 2.0 * frobenius_sq(&b_tbeta);
        metric += &graph.t().dot(&metric_tt.dot(&graph));
        metric += &graph.t().dot(&b_tbeta);
        metric += &b_tbeta.t().dot(&graph);
        if let Some(spectrum) = pinned.deflation_spectrum.as_ref() {
            // `Φ_tt − B_tt = Σₘ (λ̃ₘ − λₘ) uₘuₘᵀ` on the spectral branch.
            let mut substituted_tt = Array2::<f64>::zeros((q, q));
            for mode in 0..spectrum.raw_evals.len() {
                let shift = spectrum.cond_evals[mode] - spectrum.raw_evals[mode];
                let direction = spectrum.evecs.column(mode);
                for i in 0..q {
                    for j in 0..q {
                        substituted_tt[[i, j]] += shift * direction[i] * direction[j];
                    }
                }
            }
            substituted_metric += &graph.t().dot(&substituted_tt.dot(&graph));
        }
        lift_gram += &graph.t().dot(&graph);
        let clamp_diag = &geometry.rows[row_idx].clamp_diag;
        clamp_frobenius_sq += clamp_diag.iter().map(|value| value * value).sum::<f64>();
        let weighted_graph = Array2::from_shape_fn(graph.dim(), |(local, col)| {
            clamp_diag[local] * graph[[local, col]]
        });
        clamp_metric += &graph.t().dot(&weighted_graph);
    }
    if let Some(quotient) = sys.beta_gauge_quotient.as_ref() {
        let project = |form: &Array2<f64>| {
            let mut right = Array2::<f64>::zeros((k, k));
            for column in 0..k {
                right
                    .column_mut(column)
                    .assign(&quotient.project_complement(form.column(column)));
            }
            let mut both = Array2::<f64>::zeros((k, k));
            for row in 0..k {
                both.row_mut(row)
                    .assign(&quotient.project_complement(right.row(row)));
            }
            both
        };
        let pin = |form: &Array2<f64>| {
            let mut pinned = project(form);
            for direction in quotient.directions.iter() {
                for row in 0..k {
                    for column in 0..k {
                        pinned[[row, column]] += direction[row] * direction[column];
                    }
                }
            }
            pinned
        };
        metric = pin(&metric);
        lift_gram = pin(&lift_gram);
        substituted_metric = project(&substituted_metric);
        clamp_metric = project(&clamp_metric);
    }
    for form in [
        &mut metric,
        &mut substituted_metric,
        &mut clamp_metric,
        &mut lift_gram,
    ] {
        for row in 0..k {
            for column in (row + 1)..k {
                let mean = 0.5 * (form[[row, column]] + form[[column, row]]);
                form[[row, column]] = mean;
                form[[column, row]] = mean;
            }
        }
    }
    Ok(ExactAReducedPencilOperands {
        metric,
        substituted_metric,
        clamp_metric,
        lift_gram,
        joint_dimension,
        operator_frobenius: operator_frobenius_sq.sqrt(),
        metric_frobenius: metric_frobenius_sq.sqrt(),
        clamp_frobenius: clamp_frobenius_sq.sqrt(),
    })
}

/// Matrix-free scalar sibling of [`exact_a_reduced_classification`].
///
/// For a reduced-border direction `beta`, lift the Schur graph direction
/// `t = -A_tt^-1 A_tbeta beta` through the already classified row factors and
/// evaluate the two quadratic forms the shared exact-A classifier needs:
/// `v'B_raw v` and `v'E v`, `v = (t, beta)`.  No dense `K × K` metric is formed.
pub(crate) fn exact_a_reduced_direction_metrics(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    direction: ArrayView1<'_, f64>,
) -> Result<(f64, f64), ArrowSchurError> {
    let geometry = sys.exact_a_classification.as_ref().ok_or_else(|| {
        ArrowSchurError::SchurFactorFailed {
            reason: "exact-A reduced direction classification requires its raw B/delta/clamp carrier"
                .to_string(),
        }
    })?;
    if direction.len() != sys.k || geometry.rows.len() != sys.rows.len() {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "exact-A reduced direction classification has direction width {}, border {}, \
                 and {} carrier rows for {} system rows",
                direction.len(),
                sys.k,
                geometry.rows.len(),
                sys.rows.len(),
            ),
        });
    }

    // The reduced evidence operator is `P S P + Q Q'` when a beta gauge is
    // installed.  Classify `P beta` in the physical B/E metrics and count the
    // structural gauge pin in B at its exact unit stiffness.
    let physical_direction = match sys.beta_gauge_quotient.as_ref() {
        Some(quotient) => quotient.project_complement(direction),
        None => direction.to_owned(),
    };
    let gauge_stiffness = sys.beta_gauge_quotient.as_ref().map_or(0.0, |quotient| {
        quotient
            .directions
            .iter()
            .map(|gauge| {
                let coefficient = gauge.dot(&direction);
                coefficient * coefficient
            })
            .sum()
    });
    let beta_slice = physical_direction
        .as_slice()
        .expect("owned exact-A classification direction is contiguous");
    let mut penalty_action = vec![0.0_f64; sys.k];
    sys.penalty_matvec_add(beta_slice, &mut penalty_action);
    let mut majorizer_curvature = physical_direction
        .iter()
        .zip(penalty_action.iter())
        .map(|(&left, &right)| left * right)
        .sum::<f64>()
        + ridge_beta * physical_direction.dot(&physical_direction)
        + gauge_stiffness;
    let mut clamp_curvature = 0.0_f64;
    // #2828 — `penalty_matvec_add` applies `A_ββ = B_ββ + delta_beta`; the
    // majorizer form is `B`'s, and `E_ββ = -delta_beta` is border clamp curvature.
    if let Some(remainder) = geometry.border_remainder.as_ref() {
        if remainder.dim() != sys.k {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "exact-A reduced direction classification border remainder has width {} \
                     for border width {}",
                    remainder.dim(),
                    sys.k,
                ),
            });
        }
        let mut remainder_action = vec![0.0_f64; sys.k];
        remainder.matvec(beta_slice, &mut remainder_action);
        let remainder_form = physical_direction
            .iter()
            .zip(remainder_action.iter())
            .map(|(&left, &right)| left * right)
            .sum::<f64>();
        majorizer_curvature -= remainder_form;
        clamp_curvature -= remainder_form;
    }

    for (row_index, row) in sys.rows.iter().enumerate() {
        let q = sys.row_dims[row_index];
        let operands = &geometry.rows[row_index];
        if operands.delta_tt.dim() != (q, q)
            || operands.delta_tbeta.nrows() != q
            || operands.delta_tbeta.ncols() != operands.border_columns.len()
            || operands.clamp_diag.len() != q
        {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "exact-A reduced direction classification row {row_index} is incompatible \
                     with latent width {q} and border carrier width {}",
                    operands.border_columns.len(),
                ),
            });
        }
        let mut a_cross = Array1::<f64>::zeros(q);
        sys_htbeta_apply_row(
            sys,
            row_index,
            row,
            physical_direction.view(),
            &mut a_cross,
        );
        let mut graph = cholesky_solve_vector(htt_factors.factor(row_index), a_cross.view());
        graph.mapv_inplace(|value| -value);
        let mut b_cross = a_cross;
        for (carrier_column, &system_column) in operands.border_columns.iter().enumerate() {
            if system_column >= sys.k {
                return Err(ArrowSchurError::SchurFactorFailed {
                    reason: format!(
                        "exact-A reduced direction classification border index {system_column} \
                         exceeds width {}",
                        sys.k,
                    ),
                });
            }
            for local in 0..q {
                b_cross[local] -= operands.delta_tbeta[[local, carrier_column]]
                    * physical_direction[system_column];
            }
        }
        let b_tt = &row.htt - &operands.delta_tt;
        majorizer_curvature += graph.dot(&b_tt.dot(&graph)) + 2.0 * graph.dot(&b_cross);
        clamp_curvature += graph
            .iter()
            .zip(operands.clamp_diag.iter())
            .map(|(&value, &clamp)| clamp * value * value)
            .sum::<f64>();
    }
    Ok((majorizer_curvature, clamp_curvature))
}

pub(crate) fn factor_dense_reduced_schur(
    schur: &Array2<f64>,
    policy: ReducedSchurPolicy,
) -> Result<DenseReducedSchurFactorization, ArrowSchurError> {
    factor_dense_reduced_schur_with_exact_a(schur, policy, None)
}

pub(crate) fn factor_dense_reduced_schur_with_exact_a(
    schur: &Array2<f64>,
    policy: ReducedSchurPolicy,
    exact_a: Option<&ExactAReducedClassification>,
) -> Result<DenseReducedSchurFactorization, ArrowSchurError> {
    let newton_relative_floor = match policy {
        ReducedSchurPolicy::StrictNewton => None,
        ReducedSchurPolicy::NewtonTikhonov { relative_floor } => Some(relative_floor),
        ReducedSchurPolicy::EvidenceUnitDeflation {
            relative_floor,
            refuse_resolved_indefinite,
        } => {
            return factor_evidence_unit_deflated_schur(
                schur,
                relative_floor,
                refuse_resolved_indefinite,
                exact_a,
            );
        }
    };
    let n = schur.nrows();
    let d = jacobi_diagonal_scale(schur);
    let mut schur_scaled = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            schur_scaled[[i, j]] = schur[[i, j]] / (d[i] * d[j]);
        }
    }
    let (factor_scaled, floored_scaled) = match cholesky_lower(&schur_scaled) {
        Ok(factor) => (factor, None),
        Err(e) => {
            // #1026/#1038 — every dense reduced-Schur factorization in the SAE
            // path must honor the same opt-in spectral floor. Otherwise
            // auxiliary entry points (cross-row ordered Beta--Bernoulli
            // preconditioning) can reject the collapsed dead-atom subspace even
            // though the main direct solve would floor it and continue.
            //
            // #1803 — Newton-step callers use the Levenberg-Marquardt PD floor
            // (`spectral_pd_floored_schur`) so `Δβ` is stable. Evidence/log-det
            // callers (`unit_deflate_null_directions`) instead deflate
            // quotient/null directions to unit stiffness so they contribute the
            // ρ-independent `log 1 = 0` to the Laplace normaliser rather than a
            // ρ-dependent Occam reward for collapsed decoders.
            //
            // #2015 — this spectral floor runs on the EQUILIBRATED `schur_scaled`,
            // so `relative_floor` (a FRACTION of the operator's own max
            // eigenvalue) reads a numerically trustworthy spectrum instead of one
            // dominated by the raw column-scale spread; the floored
            // reconstruction is undone back to original units below exactly like
            // the plain factor.
            match newton_relative_floor {
                Some(relative_floor) => {
                    match spectral_pd_floored_schur(&schur_scaled, relative_floor) {
                        Some((floored, floored_factor)) => (floored_factor, Some(floored)),
                        None => {
                            return Err(ArrowSchurError::SchurFactorFailed {
                                reason: format!(
                                    "reduced Schur non-PD ({e}); spectral PD-floor declined \
                                 (no usable spectrum)"
                                ),
                            });
                        }
                    }
                }
                None => {
                    return Err(ArrowSchurError::SchurFactorFailed { reason: e });
                }
            }
        }
    };
    // Undo the equilibration exactly: L = D·L̃ (row i scaled by d_i); the
    // floored reconstruction (when present) scales back as D·S̃_floor·D.
    let mut factor = factor_scaled;
    for i in 0..n {
        let di = d[i];
        for j in 0..=i {
            factor[[i, j]] *= di;
        }
    }
    let floored_schur = floored_scaled.map(|mut floored| {
        for i in 0..n {
            for j in 0..n {
                floored[[i, j]] *= d[i] * d[j];
            }
        }
        floored
    });
    Ok(DenseReducedSchurFactorization {
        factor,
        conditioned_schur: floored_schur,
        beta_conditioning: None,
    })
}

pub(crate) fn solve_dense_reduced_system(
    schur: &Array2<f64>,
    rhs_beta: &Array1<f64>,
    options: &ArrowSolveOptions,
) -> Result<(Array1<f64>, Option<Array2<f64>>, ArrowPcgDiagnostics), ArrowSchurError> {
    let policy = ReducedSchurPolicy::newton(options.newton_schur_tikhonov_rel_floor);
    let DenseReducedSchurFactorization {
        factor,
        conditioned_schur: floored_schur,
        beta_conditioning: _,
    } = factor_dense_reduced_schur(schur, policy)?;
    if let Some(floored) = floored_schur {
        let direct = cholesky_solve_vector(&factor, rhs_beta);
        if step_inside_trust_region(direct.view(), options.trust_region.radius) {
            return Ok((direct, Some(factor), ArrowPcgDiagnostics::default()));
        }
        let identity = IdentityPreconditioner;
        let (delta, diag) =
            steihaug_dense_system(&floored, rhs_beta, &identity, &options.trust_region)?;
        return Ok((delta, Some(factor), diag));
    }
    // Ill-conditioned-but-PD Schur guard. The per-row factor checks reject
    // any single barely-PD H_tt^(i) block, but the reduced Schur complement
    //     S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)
    // accumulates the (H_tt^(i))⁻¹ contributions of every row in finite
    // precision. With many weak-but-admissible rows those terms can sum to a
    // Schur matrix whose Cholesky succeeds yet whose condition number is far
    // past the safe inversion regime, so `cholesky_solve_vector` yields an
    // inaccurate Δβ that is silently propagated to the Newton step. Apply the
    // same diagonal-ratio κ proxy used per-row to the reduced factor and treat
    // an over-threshold estimate as a Schur-stability failure: `SchurFactorFailed`
    // is already recoverable in `solve_with_lm_escalation_inner`, so this lifts
    // `ridge_beta` and re-forms a better-conditioned Schur. This guard is
    // exclusive to the dense Direct / SqrtBA path (the only caller of this
    // function); the inexact-PCG path tolerates higher κ(S) and is unaffected.
    let schur_kappa = cholesky_factor_kappa_estimate(&factor);
    if !schur_kappa.is_finite() || schur_kappa > safe_spd_kappa_max(schur.nrows()) {
        // #1026 — over-complete SAE dictionaries park surplus atoms dead
        // (β_k → 0), so the reduced Schur is PD (the Cholesky above succeeded)
        // but ILL-CONDITIONED: the dead decoder subspace carries near-zero
        // eigenvalues while the live subspace is healthy. The kappa gate's
        // concern is an inaccurate Δβ from accumulated (H_tt)⁻¹ contamination —
        // but on the dead subspace the correct Δβ IS ≈0 (those atoms have no
        // signal), so the only "inaccuracy" is in directions whose true step is
        // zero. When the spectral PD-floor is enabled (the SAE solve path),
        // clamp exactly those collapsed directions up to `floor·max(λ)` and
        // solve against the floored Schur: the live subspace keeps its EXACT
        // Newton component, the dead subspace is damped to ≈0, and κ is bounded
        // so Δβ is accurate where it matters. This is the same conditioning the
        // non-PD branch above applies; here it also covers the PD-but-ill-
        // conditioned case so the LM loop does not exhaust `ridge_β` trying to
        // (futilely) lift a fundamentally rank-deficient dead-atom subspace.
        // Without the floor (BA / non-SAE callers) the strict refusal stands.
        if let Some(relative_floor) = options.newton_schur_tikhonov_rel_floor
            && let Some((floored, floored_factor)) =
                spectral_pd_floored_schur(schur, relative_floor)
        {
            let direct = cholesky_solve_vector(&floored_factor, rhs_beta);
            if step_inside_trust_region(direct.view(), options.trust_region.radius)
            {
                return Ok((direct, Some(floored_factor), ArrowPcgDiagnostics::default()));
            }
            let identity = IdentityPreconditioner;
            let (delta, diag) =
                steihaug_dense_system(&floored, rhs_beta, &identity, &options.trust_region)?;
            return Ok((delta, Some(floored_factor), diag));
        }
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "reduced Schur complement Cholesky succeeded but is ill-conditioned \
                     (kappa_estimate={schur_kappa:e}); accumulated per-row \
                     (H_tt)⁻¹ contamination would yield an inaccurate Δβ"
            ),
        });
    }
    // Reduced-system solve. The f64 `factor` is retained and returned — its
    // diagonal is the EXACT `log|S|` the evidence path reads.
    let direct = cholesky_solve_vector(&factor, rhs_beta);
    if step_inside_trust_region(direct.view(), options.trust_region.radius) {
        return Ok((direct, Some(factor), ArrowPcgDiagnostics::default()));
    }

    // Ceres-style trust-region correction: once the dense BA solve proposes a
    // step outside the trust ball, Steihaug-CG returns the boundary point
    // without requiring a second dense factorization.
    let identity = IdentityPreconditioner;
    let (delta, diag) = steihaug_dense_system(schur, rhs_beta, &identity, &options.trust_region)?;
    Ok((delta, Some(factor), diag))
}

pub(crate) fn step_inside_trust_region(
    step: ArrayView1<'_, f64>,
    radius: f64,
) -> bool {
    !radius.is_finite() || euclidean_norm(step) <= radius
}

/// Below this row count the per-row Schur loop stays sequential: the rayon
/// fan-out (chunk dispatch + the deterministic per-chunk length-`K` reduction)
/// costs more than it saves for the handful-of-rows arrow systems that dominate
/// the non-SAE callers. Above it — the SAE LLM shape (`n` in the thousands,
/// wide border `k`) that issue #1017 names — the per-row `H_βt (H_tt)⁻¹ H_tβ x`
/// contributions are the matvec's whole cost and parallelize cleanly.
pub(crate) const SCHUR_MATVEC_PARALLEL_ROW_MIN: usize = 256;

/// Row width of the chunks the parallel Schur folds reduce into partials. It is
/// fixed rather than tied to the thread count, so the chunk sums and their fold
/// order do not depend on the pool.
pub(crate) const SCHUR_FOLD_ROW_CHUNK: usize = 64;

/// How many partials [`fold_row_chunk_partials`] keeps alive over `n_rows` rows: one
/// per pool thread, never more than there are chunks. A caller charging the
/// partials' memory before the fold charges this many.
fn fold_row_chunk_partial_count(n_rows: usize) -> usize {
    rayon::current_num_threads().clamp(1, n_rows.div_ceil(SCHUR_FOLD_ROW_CHUNK).max(1))
}

/// Reduce rows `0..n_rows` into zero-seeded partials over fixed
/// [`SCHUR_FOLD_ROW_CHUNK`]-row chunks and hand every partial to `fold` in chunk
/// order.
///
/// Partials are reduced one pool-width group of chunks at a time into reused
/// buffers, so at most `rayon::current_num_threads()` of them are alive at once.
/// Collecting every chunk's partial before folding held `n_rows /
/// SCHUR_FOLD_ROW_CHUNK` dense blocks at once: at the automatic Direct border
/// limit (`k = 2000`, 32 MiB per partial) that is about 20 GiB for 40 000 rows,
/// all of it zeroed and freed on every solve. Each buffer is re-zeroed before
/// its chunk and every chunk reduces its rows in row order, so the fold is
/// bit-identical to reducing each chunk into a fresh partial.
///
/// [`fold_row_chunk_partial_count`] is how many partials it builds.
pub(crate) fn fold_row_chunk_partials<P, E>(
    n_rows: usize,
    new_partial: impl Fn() -> P,
    zero_partial: impl Fn(&mut P) + Sync,
    reduce_row: impl Fn(usize, &mut P) -> Result<(), E> + Sync,
    fold: impl FnMut(&P),
) -> Result<(), E>
where
    P: Send,
    E: Send,
{
    fold_row_chunk_partials_at_width(
        n_rows,
        fold_row_chunk_partial_count(n_rows),
        new_partial,
        zero_partial,
        reduce_row,
        fold,
    )
}

/// [`fold_row_chunk_partials`] with `width` live partials, at least one and at most
/// one per chunk. The chunks, their row order and their fold order do not depend on
/// `width`, so every width folds the same words.
pub(crate) fn fold_row_chunk_partials_at_width<P, E>(
    n_rows: usize,
    width: usize,
    new_partial: impl Fn() -> P,
    zero_partial: impl Fn(&mut P) + Sync,
    reduce_row: impl Fn(usize, &mut P) -> Result<(), E> + Sync,
    mut fold: impl FnMut(&P),
) -> Result<(), E>
where
    P: Send,
    E: Send,
{
    use rayon::prelude::*;
    let n_chunks = n_rows.div_ceil(SCHUR_FOLD_ROW_CHUNK);
    let width = width.clamp(1, n_chunks.max(1));
    let mut partials: Vec<P> = (0..width).map(|_| new_partial()).collect();
    for first in (0..n_chunks).step_by(width) {
        let count = width.min(n_chunks - first);
        partials[..count]
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(offset, partial)| -> Result<(), E> {
                zero_partial(partial);
                let start = (first + offset) * SCHUR_FOLD_ROW_CHUNK;
                for row in start..(start + SCHUR_FOLD_ROW_CHUNK).min(n_rows) {
                    reduce_row(row, partial)?;
                }
                Ok(())
            })?;
        for partial in &partials[..count] {
            fold(partial);
        }
    }
    Ok(())
}

/// Below this border width `k` the dense `H_ββ` penalty-prologue GEMV stays
/// sequential: parallelizing a `k×k` matvec only pays once `k²` is large enough
/// to dwarf the rayon fan-out, which for the arrow callers with narrow borders
/// it never is. At the SAE LLM border (`k` in the low thousands) the `O(k²)`
/// prologue is ≈4M flops/CG-iteration and was the serial Amdahl ceiling on the
/// otherwise per-row-parallel matvec (#1017), so it crosses this threshold and
/// fans out. 512 keeps the prologue serial for every non-SAE arrow system while
/// engaging it for the wide SAE/Qwen borders the issue targets.
pub(crate) const SCHUR_PROLOGUE_PARALLEL_K_MIN: usize = 512;

/// Device-residency CPU analogue for the SAE reduced-Schur matvec (#1017).
///
/// In the production SAE joint fit the per-row cross-block factors as
/// `H_tβ^(i) = L_i P_i`, where `L_i` (`q_i × p`) is the row's local
/// assignment/coordinate Jacobian and `P_i` (`p × K`, sparse) gathers the
/// active atoms' decoder blocks (`P_i x = Σ_s φ_s · x[base_s .. base_s+p]`).
/// The reduced-Schur point-elimination contribution of one row is therefore
///
/// ```text
/// S_i x = H_βt^(i) (H_tt^(i)+ρ_t I)⁻¹ H_tβ^(i) x
///       = P_iᵀ · [ L_iᵀ (H_tt^(i)+ρ_t I)⁻¹ L_i ] · P_i x
///       = P_iᵀ G_i (P_i x),      G_i := L_iᵀ (H_tt^(i)+ρ_t I)⁻¹ L_i   (p×p).
/// ```
///
/// The block `G_i = L_iᵀ Y_i` depends only on the assembled per-row blocks and
/// the (already-computed, solve-stable) `H_tt` factor — NOT on the CG iterate
/// `x`. The generic `schur_matvec` re-walks `apply_jbeta → apply_l →
/// solve(d×d) → apply_l_t → scatter` on every CG iteration; this object **stages
/// the factors `(L_i, Y_i)` once per CG solve** (the "upload X once" residency
/// mechanism, applied on CPU to the matvec rather than a dense factorization),
/// turning each subsequent matvec into a sparse gather → two `di×p` GEMVs →
/// sparse scatter, with no per-iteration triangular solve and no operator-closure
/// re-walk. It never materialises the dense `p×p` product: `di ≪ p` for SAE
/// rows, so the factored apply is `2·support_i·p + 2·di·p` flops/row — the two
/// `di·p` GEMVs PLUS the `support_i·p` sparse gather (`P_i x`) and `support_i·p`
/// sparse scatter (`P_iᵀ prod`) — versus the dense `p²` block apply, and
/// `O(n·di·p)` memory (vs `O(n·p²)` ≈ 67 GB at the Qwen shape — the dense form
/// is OOM). For dense/full active support `support_i` can scale with the active
/// β-columns, so the gather/scatter term is NOT negligible and is counted here.
///
/// Numerically identical to the generic path up to floating-point reassociation
/// (it differentiates and accumulates the SAME quotient). It is deterministic
/// run-to-run and within the reassociation margin of the serial path, so the
/// criterion ranking across topology candidates is stable except for candidates
/// separated by less than that f64 margin, where reassociation can flip the
/// near-tie winner — it is NOT an exact no-move guarantee (#1211).
pub struct SaeResidentReducedSchur {
    /// Decoder output dimension `p` (the side length of every `G_i = L_iᵀ Y_i`).
    pub(crate) p: usize,
    /// Per-row **factored** residency: `(L_i, Y_i)`, each stored row-major as a
    /// `di × p` slab (`L_i` = local Jacobian, `Y_i = (H_tt^(i)+ρ_t I)⁻¹ L_i`).
    /// The reduced block is `G_i = L_iᵀ Y_i` (`p×p`, symmetric PSD), but it has
    /// rank ≤ `di` and `di ≪ p` for SAE rows (the per-row latent dim is 1–2
    /// while `p` is the decoder block width, ~2048). Materialising the dense
    /// `p×p` block would cost `O(n·p²)` memory (≈67 GB at the Qwen shape) and
    /// `p²` flops per matvec/row; the factored form costs `O(n·di·p)` memory and
    /// `2·support_i·p + 2·di·p` flops/row, applying `G_i v = L_iᵀ (Y_i v)`
    /// (sparse gather over `support_i` atoms → `di`-length GEMV → `p`-length
    /// GEMV → sparse scatter over `support_i` atoms). The `2·support_i·p`
    /// gather/scatter term is part of the per-row cost — for dense/full support
    /// `support_i` scales with active β-columns — and is not dropped. A row with
    /// empty active support / degenerate dims gets `di = 0` and is skipped.
    /// `(di, L_i, Y_i)` per row; `L_i`/`Y_i` are `di·p`-length row-major buffers.
    pub(crate) rows: Vec<ResidentRowFactor>,
    /// Per-row active atom support `(β-block base index, φ weight)`, shared with
    /// the assembler's [`DeviceSaePcgData`] (no re-clone of the index lists).
    pub(crate) a_phi: Arc<[Vec<(usize, f64)>]>,
    /// #1033: per-row local Jacobian `L_i` (row-major `di × p`), SHARED via `Arc`
    /// with the assembler's [`DeviceSaePcgData`] rather than copied into each
    /// `ResidentRowFactor`. The staged factor previously held its own verbatim
    /// row-major copy of `data.local_jac[row]` — a second full `O(n·di·p)` slab
    /// for zero benefit (the bytes and the `di × p` layout are identical). The
    /// matvec now reads `L_i = &self.local_jac[row]` directly; only the SOLVED
    /// factor `Y_i = (H_tt+ρI)⁻¹ L_i` (genuinely new data) stays per-row. Reads
    /// are byte-for-byte the former `rf.l` (same slab, same `r·p + c` indexing),
    /// so the matvec/preconditioner output is bit-identical.
    pub(crate) local_jac: Arc<[Vec<f64>]>,
}

/// Factored per-row residency block: `G_i = L_iᵀ Y_i` kept as its `di×p` factors
/// so the matvec never materialises the dense `p×p` product. The local Jacobian
/// factor `L_i` is NOT stored here — it is shared via
/// [`SaeResidentReducedSchur::local_jac`] (`&local_jac[row]`); only the solved
/// `Y_i` is per-row. See [`SaeResidentReducedSchur`].
pub(crate) struct ResidentRowFactor {
    /// Row latent dimension `di` (the inner contraction width). `0` ⇒ skipped.
    pub(crate) di: usize,
    /// `Y_i = (H_tt^(i)+ρ_t I)⁻¹ L_i` row-major `di × p`. Empty when `di == 0`.
    pub(crate) y: Vec<f64>,
}

impl SaeResidentReducedSchur {
    /// Stage the per-row `G_i = L_iᵀ (H_tt^(i)+ρ_t I)⁻¹ L_i` blocks once, from
    /// the SAE structure (`DeviceSaePcgData`: `p`, per-row `a_phi`, per-row
    /// row-major `local_jac` = `L_i`) and the already-factored `H_tt` slab.
    ///
    /// Returns `None` when the structure does not match (degenerate `p`, row
    /// count mismatch) so the caller falls back to the generic matvec. Row
    /// builds are independent and run under the same deterministic rayon
    /// discipline as the matvec (each `G_i` is self-contained — no cross-row
    /// reduction — so there is no ordering subtlety).
    /// `ridge_t` is NOT a parameter: it is already folded into the factored
    /// blocks `htt_factors` carry (they factor `H_tt^(i) + ridge_t·I` — see
    /// `factor_blocks`), so solving against the factor yields `(H_tt^(i)+ρ_t I)⁻¹`
    /// exactly. The residency block is a pure function of the factor and `L_i`.
    pub(crate) fn build<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        backend: &B,
    ) -> Option<Self> {
        let data = sys.device_sae_pcg.as_ref()?;
        let p = data.p;
        let n = sys.rows.len();
        if p == 0
            || sys.htbeta_dense_supplement
            || data.a_phi.len() != n
            || data.local_jac.len() != n
        {
            return None;
        }
        let empty = || ResidentRowFactor {
            di: 0,
            y: Vec::new(),
        };
        let build_row = |row: usize| -> ResidentRowFactor {
            let di = sys.row_dims[row];
            let jac = &data.local_jac[row];
            // q_i = len/p; must match the row's latent dimension di.
            if p == 0 || jac.len() != di * p || di == 0 {
                return empty();
            }
            // L_i as a (di × p) matrix (row-major in `local_jac`).
            let l_i = match ArrayView2::from_shape((di, p), jac.as_slice()) {
                Ok(v) => v.to_owned(),
                Err(_) => return empty(),
            };
            // Solve (H_tt+ρ_t I) Y = L_i for Y (di × p): one batched back-solve
            // over the p columns against the cached factor. Stage `(L_i, Y_i)`
            // — NOT the dense `p×p` product `G_i = L_iᵀ Y_i` — so storage and the
            // matvec stay `O(di·p)` instead of `O(p²)` (`di ≪ p` for SAE rows).
            let y = backend.solve_block_matrix(htt_factors.factor(row), l_i.view());
            // Flatten the SOLVED factor to a `di × p` row-major buffer (iteration
            // over a standard-layout view is row-major regardless of the source
            // strides, so the hot loop can index `r*p + c` directly). `L_i` is NOT
            // copied — the matvec reads it from the shared `local_jac` slab (it is
            // byte-for-byte `data.local_jac[row]`).
            let y_flat: Vec<f64> = y.iter().copied().collect();
            ResidentRowFactor { di, y: y_flat }
        };
        let rows: Vec<ResidentRowFactor> =
            if n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none() {
                use rayon::prelude::*;
                (0..n).into_par_iter().map(build_row).collect()
            } else {
                (0..n).map(build_row).collect()
            };
        Some(Self {
            p,
            rows,
            a_phi: data.a_phi_shared(),
            local_jac: data.local_jac_shared(),
        })
    }

    /// Accumulate one row's `S_i x = P_iᵀ G_i (P_i x) = P_iᵀ L_iᵀ Y_i (P_i x)`
    /// into `acc` (length `K`). `gather`/`prod` are caller-owned length-`p`
    /// buffers and `w` a caller-owned `≥ max_i di`-length buffer, all reused
    /// across rows to keep the hot loop allocation-free. The matvec applies the
    /// factored block in four steps: sparse gather `P_i x = Σ_s φ_s·x[base_s..]`
    /// (`support_i·p` flops), `w = Y_i·(P_i x)` (`di`-length, `di·p` flops),
    /// `prod = L_iᵀ·w` (`p`-length, `di·p` flops), and sparse scatter
    /// `acc += P_iᵀ prod` (`support_i·p` flops) — `2·support_i·p + 2·di·p`
    /// total, never the dense `p²` product. The gather/scatter `2·support_i·p`
    /// term is counted: it is not dominated by the GEMVs when the active support
    /// is wide.
    #[inline]
    pub(crate) fn row_into(
        &self,
        row: usize,
        x: &Array1<f64>,
        acc: &mut Array1<f64>,
        gather: &mut [f64],
        prod: &mut [f64],
        w: &mut [f64],
    ) {
        let rf = &self.rows[row];
        let di = rf.di;
        if di == 0 {
            return;
        }
        let p = self.p;
        let support = &self.a_phi[row];
        if support.is_empty() {
            return;
        }
        // Slice `x`/`acc` ONCE so the per-support gather/scatter (the dominant
        // `support·p` terms for wide active support) run over contiguous `f64`
        // slices — the compiler can prove unit stride and emit vectorized FMA,
        // where the former `x[base+j]`/`acc[base+j]` ndarray element indexing
        // forced a per-element strided lookup + bounds check that blocked
        // autovectorization. Every accumulation order is unchanged, so the
        // result is bit-identical to the ndarray-indexed form.
        let x_slice = x.as_slice().expect("resident matvec x must be contiguous");
        // P_i x = Σ_s φ_s · x[base_s .. base_s+p]   (length p).
        let gather = &mut gather[..p];
        for v in gather.iter_mut() {
            *v = 0.0;
        }
        for &(base, phi) in support {
            if phi == 0.0 {
                continue;
            }
            let xrow = &x_slice[base..base + p];
            for (g, &xv) in gather.iter_mut().zip(xrow) {
                *g += phi * xv;
            }
        }
        // w = Y_i · (P_i x)   (di × p GEMV → length di).  Y_i row-major di×p.
        for r in 0..di {
            let yrow = &rf.y[r * p..r * p + p];
            let mut s = 0.0_f64;
            for (&yv, &gv) in yrow.iter().zip(gather.iter()) {
                s += yv * gv;
            }
            w[r] = s;
        }
        // prod = L_iᵀ · w   (p × di GEMV → length p).  L_i row-major di×p, so
        // L_iᵀ[j,r] = L_i[r,j]; accumulate column-by-column over the di rows.
        // `L_i` is the shared `local_jac[row]` slab (#1033) — byte-for-byte the
        // former per-row `rf.l` copy.
        let l_i = &self.local_jac[row];
        let prod = &mut prod[..p];
        for v in prod.iter_mut() {
            *v = 0.0;
        }
        for r in 0..di {
            let lrow = &l_i[r * p..r * p + p];
            let wr = w[r];
            for (pj, &lj) in prod.iter_mut().zip(lrow) {
                *pj += lj * wr;
            }
        }
        // acc += P_iᵀ prod = scatter φ_s · prod into base_s blocks.
        let acc_slice = acc
            .as_slice_mut()
            .expect("resident matvec acc must be contiguous");
        for &(base, phi) in support {
            if phi == 0.0 {
                continue;
            }
            let arow = &mut acc_slice[base..base + p];
            for (a, &pv) in arow.iter_mut().zip(prod.iter()) {
                *a += phi * pv;
            }
        }
    }

    /// Max row latent dim `di` across resident rows — the size of the `w`
    /// scratch the matvec needs for the inner `Y_i·(P_i x)` GEMV.
    pub(crate) fn max_di(&self) -> usize {
        self.rows.iter().map(|r| r.di).max().unwrap_or(0)
    }
}

/// Reduced-Schur matvec `out = S·x` with an optional pre-staged SAE residency
/// operator. When `resident` is `Some`, the per-row point-elimination term is
/// applied through the resident `p×p` blocks (#1017 CPU residency); otherwise it
/// falls back to the generic per-row `apply → solve → transpose` path. Both
/// routes accumulate the SAME reduced operator
/// `S = H_ββ + ρ_β I − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)`.
pub(crate) fn schur_matvec<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    x: &Array1<f64>,
    out: &mut Array1<f64>,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
) {
    // `steihaug_cg` reuses one output buffer across iterations and requires
    // `matvec` to ASSIGN every entry of `out` (the contract `dense_matvec`
    // upholds). This routine builds `S·x` purely by accumulation
    // (`penalty_matvec_add`, `out[a] += ridge·x`, `out[a] -= neg_contrib`), so it
    // MUST clear `out` first. Without this, iteration n>0 returns `S·x` plus the
    // previous call's `S·p`, the PCG solves a corrupted reduced system, and the
    // resulting Newton step is inconsistent with the assembled gradient
    // (g·δ ≈ 0 — a non-descent direction that defeats the line search).
    out.fill(0.0);
    let k = sys.k;
    // Top-level (not nested in a rayon worker) and big enough to amortize the
    // fan-out: the single gate that authorizes BOTH the dense penalty-prologue
    // GEMV and the per-row point-elimination loop to go parallel. The topology
    // race fans candidates with `run_topology_race_parallel`, so inside a worker
    // both stay sequential (no nested-rayon oversubscription).
    let parallel =
        sys.rows.len() >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
    // Route the penalty-side (H_ββ + ridge·I) x product through the prologue:
    // no Arc-clone hot-path cost when penalty_op is None (falls back to hbb
    // inline); the dense fallback fans across cores at the wide SAE border (#1017).
    {
        let x_slice = x.as_slice().expect("x must be contiguous");
        let out_slice = out.as_slice_mut().expect("out must be contiguous");
        sys.penalty_ridge_prologue_into(x_slice, ridge_beta, out_slice, parallel);
    }
    // The reduced-Schur point-elimination term: `out -= Σ_i H_βt^(i) (H_tt^(i))⁻¹
    // H_tβ^(i) x`. Each row contributes an independent length-`K` vector, so for
    // the SAE LLM shape (#1017) this is the matvec's whole cost and is
    // embarrassingly parallel — reduced below through the deterministic pairwise
    // tree (see the block-fold comment) rather than a chunk-order fold.
    let p = resident.map(|r| r.p).unwrap_or(0);
    // #2228 determinism: the per-row length-`k` contributions
    // (`Σ_i H_βt^(i)(H_tt^(i))⁻¹ H_tβ^(i) x`) are reduced through the length-only
    // pairwise tree so the result is bit-identical across thread count AND to the
    // sequential fold — parallel and nested-serial evaluation agree to the last
    // bit, removing the #1017/#1211 chunk-reassociation margin that let the
    // criterion ranking depend on the driver. The tree self-serializes below
    // `BASE_CHUNK` rows (a base block is folded directly with no `rayon::join`),
    // so small systems and nested topology-race calls stay single-threaded
    // without a separate branch that could associate the round-off differently.
    // The resident path gathers → factored `di×p` GEMVs → scatter; the direct
    // path does a per-row block solve — both ADD their row's contribution into a
    // block-local accumulator, so splitting the row sum across the tree is exact.
    let n_rows = sys.rows.len();
    let contribution = gam_linalg::pairwise_reduce::par_deterministic_block_fold(
        n_rows,
        |range: core::ops::Range<usize>| {
            let mut acc = Array1::<f64>::zeros(k);
            if let Some(res) = resident {
                let mut gather = vec![0.0_f64; p];
                let mut prod = vec![0.0_f64; p];
                let mut w = vec![0.0_f64; res.max_di()];
                for i in range {
                    res.row_into(i, x, &mut acc, &mut gather, &mut prod, &mut w);
                }
            } else {
                let mut local = Array1::<f64>::zeros(sys.d);
                for i in range {
                    schur_matvec_row_into(sys, htt_factors, x, backend, i, &mut local, &mut acc);
                }
            }
            acc
        },
        |mut a: Array1<f64>, b: Array1<f64>| {
            a += &b;
            a
        },
    );
    if let Some(acc) = contribution {
        for a in 0..k {
            out[a] -= acc[a];
        }
    }
}

/// #1017: the reduced-Schur operator `v ↦ S·v` staged ONCE per criterion
/// evaluation and reused across EVERY shifted / warm-started solve of the
/// rational-logdet (and SLQ) ladder — the widened-lifetime residency the #1017
/// device design calls for.
///
/// The rational-logdet criterion (`matrix_free_arrow_evidence_log_det_surrogate`)
/// walks SEVERAL shift ladders inside ONE evaluation: the `λ_max` power iteration
/// (`reduced_schur_lambda_max`), the pilot / deflation-derived plan build
/// (`rational_reduced_schur_plan_derived`), the value [`RationalLogdetPlan::
/// evaluate`], and the `(probes, S⁻¹·probes)` gradient bundle
/// (`reduced_schur_inverse_probe_solves`). Each formerly re-captured its own
/// inline `schur_matvec` closure over `(sys, htt_factors, ρ_β, backend,
/// resident)`. On CPU those captures are free; on the device lane they are the
/// per-solve FLATTEN — every ladder would re-marshal and re-upload the
/// ridge-independent operands (the factored `H_tt` slab, the framed `G ⊗ W`, the
/// dense per-row cross blocks) that are INVARIANT across the whole evaluation.
///
/// This object is the single operator every ladder borrows: the invariant state
/// (`sys`, the factored `H_tt` slab, the `ρ_β` border, the pre-staged CPU
/// [`SaeResidentReducedSchur`] frame, and — when engaged — a device-resident
/// [`GpuSchurMatvec`] whose per-row factors upload ONCE) lives for the whole
/// evaluation, so a shifted solve reuses the resident operator instead of
/// re-staging it. Every `apply` accumulates the SAME reduced operator
/// `S = (H_ββ + ρ_β I) − Σ_i H_βt^(i)(H_tt^(i)+ρ_t I)⁻¹H_tβ^(i)` regardless of
/// lane. With `gpu_matvec == None` (every current construction) the result is
/// byte-for-byte the pre-context inline `schur_matvec` closure; the `gpu_matvec`
/// seam is where a device operator, built once per evaluation, is threaded through
/// the ladder (the reported #1017 next increment).
pub(crate) struct ReducedSchurOperator<'a, B: BatchedBlockSolver + Sync> {
    sys: &'a ArrowSchurSystem,
    htt_factors: &'a ArrowFactorSlab,
    ridge_beta: f64,
    backend: &'a B,
    resident: Option<&'a SaeResidentReducedSchur>,
    gpu_matvec: Option<&'a GpuSchurMatvec>,
}

impl<'a, B: BatchedBlockSolver + Sync> ReducedSchurOperator<'a, B> {
    /// The CPU/host operator — the byte-identical default. Every shifted solve in
    /// the evaluation reuses the same pre-staged `resident` frame (or the generic
    /// per-row `apply → solve → transpose` when `resident` is `None`).
    pub(crate) fn new(
        sys: &'a ArrowSchurSystem,
        htt_factors: &'a ArrowFactorSlab,
        ridge_beta: f64,
        backend: &'a B,
        resident: Option<&'a SaeResidentReducedSchur>,
    ) -> Self {
        Self {
            sys,
            htt_factors,
            ridge_beta,
            backend,
            resident,
            gpu_matvec: None,
        }
    }

    /// Attach a device-resident [`GpuSchurMatvec`] (built ONCE per evaluation) so
    /// the whole ladder applies `S·v` on device without a per-solve re-upload.
    /// #1017 next increment: the caller that owns the device operand upload builds
    /// the operator once and calls this; until then every construction is CPU
    /// (`gpu_matvec == None`), so the lane stays byte-identical.
    pub(crate) fn with_gpu_matvec(mut self, gpu_matvec: Option<&'a GpuSchurMatvec>) -> Self {
        self.gpu_matvec = gpu_matvec;
        self
    }

    /// `out = S·x`. Both lanes CLEAR and fully assign `out`, so a fresh zeroed
    /// buffer per apply is correct (and the shift-ladder CG contract is upheld).
    #[inline]
    pub(crate) fn apply_into(&self, x: &Array1<f64>, out: &mut Array1<f64>) {
        if let Some(quotient) = self.sys.beta_gauge_quotient.as_ref() {
            // Evidence operator on the quotient: `P S P + Q Q^T`.  Apply the
            // original reduced Schur only to `P x`, project its result once more,
            // then add the unit Faddeev--Popov pin. The same arithmetic is used by
            // dense `pin_reduced_schur`, so SLQ/rational-logdet values and dense
            // Cholesky values represent the identical operator.
            let projected_x = quotient.project_complement(x.view());
            if let Some(gpu) = self.gpu_matvec {
                gpu(&projected_x, out);
            } else {
                schur_matvec(
                    self.sys,
                    self.htt_factors,
                    self.ridge_beta,
                    &projected_x,
                    out,
                    self.backend,
                    self.resident,
                );
            }
            let mut projected_out = quotient.project_complement(out.view());
            for direction in quotient.directions.iter() {
                projected_out.scaled_add(direction.dot(x), direction);
            }
            out.assign(&projected_out);
        } else {
            if let Some(gpu) = self.gpu_matvec {
                gpu(x, out);
            } else {
                schur_matvec(
                    self.sys,
                    self.htt_factors,
                    self.ridge_beta,
                    x,
                    out,
                    self.backend,
                    self.resident,
                );
            }
        }

        if let Some(conditioning) = self.sys.exact_a_reduced_conditioning.as_ref() {
            assert_eq!(conditioning.directions.len(), conditioning.shifts.len());
            for (direction, &shift) in conditioning
                .directions
                .iter()
                .zip(conditioning.shifts.iter())
            {
                assert_eq!(direction.len(), self.sys.k);
                out.scaled_add(shift * direction.dot(x), direction);
            }
        }
    }

    /// `S·v` into a fresh length-`k` vector — the shift-ladder matvec-closure form
    /// (`|v: ArrayView1| op.apply(v)`). Byte-for-byte the inline
    /// `let x = v.to_owned(); schur_matvec(…, &x, &mut zeros(k), …)` it replaces.
    #[inline]
    pub(crate) fn apply(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let x = v.to_owned();
        let mut out = Array1::<f64>::zeros(self.sys.k);
        self.apply_into(&x, &mut out);
        out
    }

    /// `S·x` into a fresh vector from an already-owned `&Array1` (no redundant copy
    /// of a vector the caller already owns) — the power-iteration / CG-solve form.
    #[inline]
    pub(crate) fn apply_owned(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.sys.k);
        self.apply_into(x, &mut out);
        out
    }
}

/// Matrix-free reduced-Schur log-determinant `log|S|` via Stochastic Lanczos
/// Quadrature on the exact `schur_matvec` apply `v ↦ S·v`, where
/// `S = (H_ββ + ρ_β I) − Σ_i H_βt^(i)(H_tt^(i)+ρ_t I)⁻¹H_tβ^(i)` is the SPD
/// reduced Schur. **The dense `k×k` `S` is NEVER formed.**
///
/// This is the memory-matrix-free evidence path for the massive-K manifold SAE.
/// The dense evidence routes assemble `S` explicitly (`O(k²)` ≈ 8 GB at the
/// K=32k border) and Cholesky-factor it (`O(k³/3)`) purely to read `Σ 2·log Lᵢᵢ`;
/// that dense assembly + factor is the massive-K wall (both dense evidence
/// routes REFUSE above the in-core budget). Here peak memory is `O(k)` — the SLQ
/// Rademacher probe and Lanczos basis vectors — and the cost is
/// `O(num_probes·lanczos_steps · matvec)`, each matvec the same `O(n·d·k)`
/// reduced-Schur apply the PCG hot loop already runs. Deterministic for a fixed
/// `(sys, htt_factors, ρ_β, resident, num_probes, lanczos_steps, seed)` so the
/// REML evidence outer loop stays reproducible.
///
/// `htt_factors` are the per-row `(H_tt^(i)+ρ_t I)` Cholesky factors; `resident`
/// is the optional pre-staged SAE residency operator (`None` for the framed /
/// closure `H_tβ` path). SLQ is an ESTIMATE; callers that need the exact dense
/// log-det at small `k` must stay on the dense route.
///
/// Crate-internal because the `resident` parameter carries the `pub(crate)`
/// [`SaeResidentReducedSchur`] operator; cross-crate callers use the
/// [`matrix_free_arrow_evidence_log_det_surrogate`] entry, which stages residency
/// internally and exposes no crate-private type.
pub(crate) fn slq_reduced_schur_log_det<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    evidence_policy: ArrowEvidencePolicy,
    num_probes: usize,
    lanczos_steps: usize,
    seed: u64,
) -> Result<SlqLogDet, ArrowSchurError> {
    let k = sys.k;
    // Stage the reduced-Schur operator ONCE; every probe/Lanczos apply reuses the
    // pre-staged residency (no per-apply operator re-capture). The probes fan
    // across rayon workers (in `slq_logdet`), and `schur_matvec`'s own row
    // parallelism is guarded off inside a worker, so there is no nested
    // oversubscription. When `gpu_matvec` is `Some` (the #1017 Phase-3 device
    // seam, built once for the whole evidence evaluation), EVERY Rademacher-probe
    // Lanczos apply runs through the single resident device `S·v`; when `None`
    // the byte-identical CPU `schur_matvec` lane is taken.
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    // The evidence log|S| must obey the SAME conditioning convention as the dense
    // reduced-Schur factor (#2308). Under `UnitDeflation` a collapsed / near-null
    // decoder direction is pinned to unit stiffness (`ln 1 = 0`), so the SLQ
    // estimate uses the unit-deflated spectral function `φ(θ)=θ≥floor ? ln θ : 0`
    // instead of the plain `ln` (which would floor a sub-null Ritz value to
    // `RITZ_LN_FLOOR`, contributing `≈ −690` per collapsed direction and a
    // ρ-dependent Occam reward). `Strict` / `PositiveDefinite` keep the plain SPD
    // estimator — they never form an undamped evidence with nulls.
    match evidence_policy {
        ArrowEvidencePolicy::UnitDeflation { relative_floor } => Ok(slq_logdet_unit_deflated(
            k,
            |v| op.apply(v),
            num_probes,
            lanczos_steps,
            seed,
            relative_floor,
        )
        .as_logdet()),
        // #2515 — a negative Ritz value is a Rayleigh quotient of raw A, not a
        // saddle verdict. Lift each Ritz direction and ask the same typed
        // B-metric/clamp-basin classifier as the dense and direct-arrow routes.
        ArrowEvidencePolicy::UnitDeflationRefusingIndefinite {
            relative_floor: _,
        } => {
            if sys.exact_a_classification.is_none() {
                return Err(ArrowSchurError::SchurFactorFailed {
                    reason: "matrix-free exact-A evidence policy requires the raw B/delta/clamp \
                             classification carrier"
                        .to_string(),
                });
            }
            slq_logdet_exact_a_classified(
                k,
                |v| op.apply(v),
                |direction| {
                    exact_a_reduced_direction_metrics(
                        sys,
                        htt_factors,
                        ridge_beta,
                        direction,
                    )
                    .map_err(|error| error.to_string())
                },
                num_probes,
                lanczos_steps,
                seed,
            )
            .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })
        }
        ArrowEvidencePolicy::Strict | ArrowEvidencePolicy::PositiveDefinite => {
            Ok(slq_logdet(k, |v| op.apply(v), num_probes, lanczos_steps, seed))
        }
    }
}

/// #1017 Phase-3: build the reduced-Schur device matvec ONCE for a matrix-free
/// evidence log-det evaluation, so the whole rational-logdet + SLQ ladder applies
/// `S·v` through a single device-resident operator (uploaded / pre-factored once)
/// rather than re-capturing the CPU `schur_matvec` per probe / shifted solve. The
/// PCG numerics are identical whether the matvec runs on host or device (same
/// reduced Schur operator, same f64 accumulation), so engaging it changes only
/// where the `Σ_i H_βt(H_tt)⁻¹H_tβ` flops execute.
///
/// Same admission contract as the PCG matvec offload ([`maybe_inject_gpu_schur_matvec`]):
/// declines (returns `None`, so every apply stays on the byte-identical CPU lane)
/// when streaming is present, the work predicate rejects
/// the shape, or no live device is present. `apply_budget` is the amortising apply
/// count for the shape predicate — the reduced-Schur matvec is `O(n·d·k)` per
/// apply and the evidence ladder runs that apply across every probe / Lanczos /
/// shifted-CG step, so a large budget is the honest amortisation the offload
/// break-even is measured against.
pub(crate) fn maybe_build_evidence_gpu_matvec(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
    apply_budget: usize,
) -> Result<Option<GpuSchurMatvec>, ArrowSchurError> {
    // A caller-supplied operator (threaded through `options.gpu_matvec`) already
    // owns its residency; the caller passes it directly, so never double-build.
    if options.gpu_matvec.is_some() {
        return Ok(None);
    }
    if options.streaming_chunk_size.is_some() {
        return Ok(None);
    }
    // Size gate BEFORE the device probe (startup-tax ordering): at the most
    // permissive floor any production policy can carry, the predicate rejects
    // exactly the shapes every reachable policy rejects, so those skip runtime
    // availability resolution (whose first call creates a CUDA primary context on
    // every GPU). An admitted shape probes exactly as the PCG seam does, and the
    // probed device's calibrated policy then decides.
    if !gam_gpu::GpuDispatchPolicy::reduced_schur_matvec_admissible_under_any_policy(
        sys.rows.len(),
        sys.k,
        sys.d,
        apply_budget.max(1),
    ) {
        return Ok(None);
    }
    let Some(runtime) = gam_gpu::device_runtime::GpuRuntime::resolve(options.gpu_policy)
        .map_err(|error| ArrowSchurError::SchurFactorFailed {
            reason: format!("evidence GPU runtime resolution failed: {error}"),
        })?
    else {
        return Ok(None);
    };
    if !runtime.policy().reduced_schur_matvec_should_offload(
        sys.rows.len(),
        sys.k,
        sys.d,
        apply_budget.max(1),
    ) {
        return Ok(None);
    }
    // #1017: framed matrix-free system with resident device operands — prefer the
    // device-resident DETERMINISTIC reduced-Schur apply (upload operands once,
    // cross only x/out per apply, atomics-free so the SLQ log|S| determinism
    // contract holds) over the CPU row-procedural closure `gpu_schur_matvec_backend`
    // returns for `htbeta_matvec` systems. Declines (no device / shape / non-PD at
    // this ridge) fall through to the backend/CPU path. Non-Linux/CPU: this always
    // returns `None` (no `device_sae_pcg`), so the lane is byte-identical.
    // `Unavailable` is the device saying "not this shape/config", which is a
    // DECLINE and not a fault: every other exit from this function reports a
    // decline as `Ok(None)`, the CPU lane.
    // Surfacing it as an error made a host WITH a GPU fail where a CPU-only host
    // returned `Ok(None)` at the runtime probe and passed. Genuine faults
    // (`RidgeBumpRequired`, `SchurFactorFailed`) still surface.
    if sys.device_sae_pcg.is_some() {
        match crate::gpu_kernels::arrow_schur::build_framed_resident_evidence_matvec(
            sys,
            ridge_t,
            ridge_beta,
            apply_budget.max(1),
        ) {
            Ok(Some(matvec)) => return Ok(Some(matvec)),
            Ok(None) => {}
            Err(crate::gpu_kernels::arrow_schur::ArrowSchurGpuFailure::Unavailable) => {
                log::trace!("resident evidence matvec build: device unavailable; CPU matvec");
            }
            Err(failure) => {
                return Err(device_failure_as_arrow_error(
                    "resident evidence matvec build",
                    failure,
                ));
            }
        }
    }
    match crate::gpu_kernels::arrow_schur::gpu_schur_matvec_backend(sys, ridge_t, ridge_beta) {
        Ok(matvec) => Ok(Some(matvec)),
        Err(crate::gpu_kernels::arrow_schur::ArrowSchurGpuFailure::Unavailable) => Ok(None),
        Err(failure) => Err(device_failure_as_arrow_error("evidence matvec build", failure)),
    }
}

/// Fixed configuration for the #2080 rational-surrogate evidence lane: the probe
/// count, seeds, quadrature/CG tolerances, and derived-rank deflation schedule the
/// [`SurrogateLaneState`] plan is (re)built with. The caller (the SAE streaming
/// criterion) supplies these once; `deflation_target_std_err_rel` is the derived
/// bar `0.1 · STALL_REL_TOL` (see `rational_reduced_schur_plan_derived`), or `+∞`
/// where a caller consumes no value bar. The support LAML lane validates its
/// certified point on independent probes instead (#2933 F28). The
/// deflation rank has no requested ceiling: the ladder may climb to the
/// operator's own dimension (#2731).
#[derive(Clone)]
pub struct SurrogateLaneConfig {
    pub num_probes: usize,
    pub seed: u64,
    pub rel_tol: f64,
    pub cg_rel_tol: f64,
    pub deflation_subspace_iters: usize,
    pub deflation_target_std_err_rel: f64,
}

/// Per-outer-solve state for the #2080 rational-surrogate evidence lane. Holds
/// the FROZEN derived-rank plan — probes, bracket-centred quadrature, and Hutch++
/// `Q`, all fixed once at the entry ρ so value and gradient stay a single
/// functional across the ρ sweep — plus the config to (re)build it when the
/// reduced-Schur dimension changes (a basin mutation between outer solves). An
/// evaluation whose caller admits the dense reduced Schur takes the exact
/// log-determinant instead and freezes nothing (#2731).
/// Threaded as `Option<&mut _>` through the streaming criterion; `None` keeps the
/// bit-identical SLQ path.
pub struct SurrogateLaneState {
    plan: Option<RationalLogdetPlan>,
    cfg: SurrogateLaneConfig,
    /// When set, the next matrix-free evidence eval also computes the shared
    /// `(probes, S⁻¹·probes)` bundle for EFS/MacKay proposal traces and stashes
    /// it in `inverse_probes`. It is never an outer gradient artifact: the fixed
    /// rational value's derivative is `logdet_derivative_bundle` below.
    request_inverse_probes: bool,
    /// The last-computed shared bundle: the FROZEN plan's probes `v_j` and their
    /// `S⁻¹ v_j` (t = 0) solves at the current operator. One bundle drives every
    /// selected-inverse trace `tr(S⁻¹·M) ≈ (1/m)Σ_j (S⁻¹v_j)ᵀ(M v_j)` off the
    /// same frozen raw probes as the value plan. This is useful for EFS trace
    /// proposals but is not the derivative of the shifted rational value.
    inverse_probes: Option<(Vec<Array1<f64>>, Vec<Array1<f64>>)>,
    /// Request/stash the lossless weighted derivative representation emitted by
    /// the next rational value evaluation.  Unlike `inverse_probes`, this is the
    /// derivative of the fixed rational surrogate itself (all shifted solves and
    /// frozen-Q columns), and is the only bundle admissible for its outer
    /// gradient.
    request_logdet_derivative_bundle: bool,
    logdet_derivative_bundle: Option<RationalLogdetDerivativeBundle>,
    /// The band directions the most recent derivative bundle carries (#2933 F07): the
    /// reduced-Schur directions the dense exact-A lane priced at the metric's own curvature
    /// `μ̃ = 1`. Along each, the bundle contracts `wᵀ dS_A w` while the value's derivative is
    /// `wᵀ dY w`, so the channel `Σ_band wᵀ(dY − dS_A)w` is missing from every derivative
    /// taken off that bundle. Empty when the bundle came from any other lane, none of which
    /// classifies a pencil.
    logdet_derivative_band_directions: Vec<ReducedBandDirection>,
    /// The previous ρ's `S⁻¹ v_j` solves, kept as the CG warm-start for the next
    /// bundle solve. `S⁻¹` is smooth in ρ, so a neighbouring-ρ solution is a near
    /// seed (common-random-numbers reuse — the discipline that makes the
    /// surrogate's shifted ladder cheap); the converged solve is unchanged to
    /// `cg_rel_tol`, only its iteration count drops. Cleared when the plan rebuilds
    /// (basin border change ⇒ the old-dim seeds are meaningless).
    warm_inverse_probes: Option<Vec<Array1<f64>>>,
}

impl SurrogateLaneState {
    /// A lane with no plan yet — the first evaluation builds and freezes it.
    pub fn new(cfg: SurrogateLaneConfig) -> Self {
        Self {
            plan: None,
            cfg,
            request_inverse_probes: false,
            inverse_probes: None,
            request_logdet_derivative_bundle: false,
            logdet_derivative_bundle: None,
            logdet_derivative_band_directions: Vec::new(),
            warm_inverse_probes: None,
        }
    }

    /// The frozen plan, once built (for the gradient lane, which contracts
    /// against the SAME `Q` the value used).
    pub fn plan(&self) -> Option<&RationalLogdetPlan> {
        self.plan.as_ref()
    }

    /// Ask the next matrix-free evidence eval to also emit the shared
    /// `(probes, S⁻¹·probes)` bundle. Clears any stale bundle so a failed or
    /// skipped eval cannot hand back last call's solves.
    pub fn request_inverse_probes(&mut self) {
        self.request_inverse_probes = true;
        self.inverse_probes = None;
    }

    /// Take the shared bundle produced by the most recent eval, if requested and
    /// computed. Consumes it so a later gradient read cannot reuse stale solves.
    pub fn take_inverse_probes(&mut self) -> Option<(Vec<Array1<f64>>, Vec<Array1<f64>>)> {
        self.request_inverse_probes = false;
        self.inverse_probes.take()
    }

    /// Ask the next rational value evaluation to retain its complete weighted
    /// derivative representation. Clears stale output eagerly so a failed value
    /// cannot be paired with a previous operator's gradient.
    pub fn request_logdet_derivative_bundle(&mut self) {
        self.request_logdet_derivative_bundle = true;
        self.logdet_derivative_bundle = None;
        self.logdet_derivative_band_directions.clear();
    }

    /// The band directions the most recent derivative bundle carries at unit curvature,
    /// along which its derivative is missing the channel `wᵀ(dY − dS_A)w` (#2933 F07). The
    /// count is the slice's length; empty unless the dense exact-A lane emitted the bundle.
    pub fn logdet_derivative_band_directions(&self) -> &[ReducedBandDirection] {
        &self.logdet_derivative_band_directions
    }

    /// Consume the derivative representation produced by the most recent
    /// requested rational value evaluation.
    pub fn take_logdet_derivative_bundle(&mut self) -> Option<RationalLogdetDerivativeBundle> {
        self.request_logdet_derivative_bundle = false;
        self.logdet_derivative_bundle.take()
    }
}

/// Split arrow-Schur evidence `log|H| = Σ log|H_tt| + log|S|` where the reduced
/// Schur term is estimated by the #2080 rational surrogate rather than SLQ, on
/// ONE shared factorization:
///
/// - `lane = None` runs the identical `slq_reduced_schur_log_det` path — a
///   bit-for-bit fallback so a caller that has not opted in is unchanged.
/// - `lane = Some(state)` builds (or, when the reduced-Schur dimension is
///   unchanged, reuses) the frozen derived-rank [`RationalLogdetPlan`] and
///   evaluates it against the current operator. The plan's `Q`/probes/quadrature
///   are fixed at first build, so only the matrix-free `S·v` apply moves with ρ —
///   the value and its `RationalLogdetPlan::directional_derivative` gradient
///   remain one functional.
///
/// Returns `(log_det_tt, log_det_schur)`; the caller adds them for the evidence.
pub fn matrix_free_arrow_evidence_log_det_surrogate(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
    slq_num_probes: usize,
    slq_lanczos_steps: usize,
    slq_seed: u64,
    lane: Option<&mut SurrogateLaneState>,
) -> Result<(f64, f64), ArrowSchurError> {
    let (log_det_tt, log_det_schur, _factors, _clamp_basin, _band) = matrix_free_arrow_evidence_log_det_surrogate_core(
        sys,
        ridge_t,
        ridge_beta,
        options,
        slq_num_probes,
        slq_lanczos_steps,
        slq_seed,
        lane,
        false,
    )?;
    Ok((log_det_tt, log_det_schur))
}

/// One matrix-free evidence value together with the exact row geometry that
/// produced it. The reduced-Schur derivative bundle emitted through
/// [`SurrogateLaneState`] is only meaningful when lifted through these same row
/// factors and cross blocks; retaining a different operator's factor cache
/// silently differentiates a different log determinant.
pub struct MatrixFreeArrowEvidenceEvaluation {
    pub log_det_tt: f64,
    pub log_det_schur: f64,
    pub factor_cache: ArrowFactorCache,
    /// Directions of this evaluation's operator that the shared exact-A classifier
    /// priced at their clamp basin: the row blocks' and the reduced Schur's, counted
    /// where the classifier returned. Zero under a majorizer policy, which has no
    /// classifier. A basin price moves with the fitted state, so a consumer that
    /// differentiates the value with the conditioning held fixed reads this to know
    /// whether it may (#2933 F27).
    pub exact_a_clamp_basin_directions: usize,
    /// The reduced-Schur directions the dense exact-A lane priced at the metric's own
    /// curvature: pencil directions inside their band edge and basin directions inside
    /// their floor (#2933 F07). Empty on every other route, none of which classifies a
    /// pencil.
    pub exact_a_reduced_band_directions: Vec<ReducedBandDirection>,
}

/// Which spectrum of the dense exact-A lane a [`ReducedBandDirection`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReducedBandOrigin {
    /// A pencil direction `S_A w = μ Y w` with `|μ|` at or under its band edge.
    Pencil,
    /// A direction of the priced basin `C = W_NᵀS_A W_N + W_NᵀE W_N` whose curvature lies
    /// inside its floor once the clamp is restored.
    Basin,
}

/// One reduced-Schur direction the dense exact-A lane priced at the metric's own
/// curvature `μ̃ = 1` (#2933 F07), with the operands of its verdict.
#[derive(Debug, Clone)]
pub struct ReducedBandDirection {
    /// The border direction `w`, normalized in the pencil metric: `wᵀYw = 1`.
    pub direction: Array1<f64>,
    /// `μ` for a pencil direction, the basin curvature `κ` for a basin direction.
    pub curvature: f64,
    /// The edge [`exact_a_band_edge`] returned, which `|curvature|` did not clear.
    pub edge: f64,
    /// The numerical resolution [`exact_a_pencil_resolution`] of `curvature`.
    pub resolution: f64,
    /// `s = wᵀ(Φ − B_raw)w` over the spectral pins; zero for a basin direction.
    pub substituted_stiffness: f64,
    pub origin: ReducedBandOrigin,
}

impl MatrixFreeArrowEvidenceEvaluation {
    #[must_use]
    pub fn log_det(&self) -> f64 {
        self.log_det_tt + self.log_det_schur
    }
}

/// Gradient-bearing form of
/// [`matrix_free_arrow_evidence_log_det_surrogate`]. Value, rational derivative,
/// and row factors are emitted by one factorization; consumers therefore cannot
/// pair the derivative of one reduced operator with another operator's row
/// elimination geometry.
///
/// `dense_reduced_schur_admitted` is the caller's memory planner's verdict on the
/// dense `k × k` reduced Schur. When it admits the block, the lane takes the exact
/// log-determinant off one eigendecomposition of the operator, and the derivative
/// bundle is the exact `tr(S⁻¹·D)` representation; otherwise the frozen rational
/// surrogate runs (#2731).
pub fn matrix_free_arrow_evidence_evaluation(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
    slq_num_probes: usize,
    slq_lanczos_steps: usize,
    slq_seed: u64,
    lane: &mut SurrogateLaneState,
    dense_reduced_schur_admitted: bool,
) -> Result<MatrixFreeArrowEvidenceEvaluation, ArrowSchurError> {
    if ridge_t != 0.0 || ridge_beta != 0.0 {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "gradient-bearing evidence must be undamped, got ridge_t={ridge_t:e}, \
                 ridge_beta={ridge_beta:e}"
            ),
        });
    }
    let (
        log_det_tt,
        log_det_schur,
        factorization,
        reduced_clamp_basin_directions,
        exact_a_reduced_band_directions,
    ) = matrix_free_arrow_evidence_log_det_surrogate_core(
            sys,
            ridge_t,
            ridge_beta,
            options,
            slq_num_probes,
            slq_lanczos_steps,
            slq_seed,
            Some(lane),
            dense_reduced_schur_admitted,
        )?;
    let reduced_clamp_basin_directions =
        reduced_clamp_basin_directions.ok_or_else(|| ArrowSchurError::SchurFactorFailed {
            reason: "matrix-free evidence evaluation surfaced no reduced-Schur classifier \
                     verdicts although it ran on a surrogate lane"
                .to_string(),
        })?;
    let exact_a_clamp_basin_directions =
        factorization.clamp_basin_directions + reduced_clamp_basin_directions;
    let factor_cache = ArrowFactorCache {
        htt_factors: factorization.factors,
        htt_factors_undamped: ArrowUndampedFactors::SameAsDamped,
        schur_factor: None,
        schur_factor_is_undamped: true,
        beta_schur_conditioning: None,
        joint_hessian_log_det: Some(log_det_tt + log_det_schur),
        solver_mode: options.mode,
        ridge_t,
        ridge_beta,
        htbeta: ArrowHtbetaCache::from_system(sys)?,
        d: sys.d,
        row_dims: Arc::clone(&sys.row_dims),
        row_offsets: Arc::clone(&sys.row_offsets),
        k: sys.k,
        manifold_mode_fingerprint: sys.manifold_mode_fingerprint,
        row_hessian_fingerprint: sys.current_row_hessian_fingerprint(),
        pcg_diagnostics: ArrowPcgDiagnostics::default(),
        gauge_deflated_directions: factorization.gauge_deflated_directions,
        deflated_row_directions: factorization.deflated_row_directions.into(),
        deflation_row_spectra: factorization.deflation_row_spectra.into(),
        beta_gauge_quotient: sys.beta_gauge_quotient.clone(),
    };
    Ok(MatrixFreeArrowEvidenceEvaluation {
        log_det_tt,
        log_det_schur,
        factor_cache,
        exact_a_clamp_basin_directions,
        exact_a_reduced_band_directions,
    })
}

fn matrix_free_arrow_evidence_log_det_surrogate_core(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    options: &ArrowSolveOptions,
    slq_num_probes: usize,
    slq_lanczos_steps: usize,
    slq_seed: u64,
    lane: Option<&mut SurrogateLaneState>,
    dense_reduced_schur_admitted: bool,
) -> Result<
    (
        f64,
        f64,
        ArrowBlockFactorization,
        Option<usize>,
        Vec<ReducedBandDirection>,
    ),
    ArrowSchurError,
> {
    let backend = CpuBatchedBlockSolver;
    let factorization = factor_blocks_for_system(
        sys,
        ridge_t,
        options.evidence_policy,
        &backend,
        options.gpu_policy,
    )?;
    let htt_factors = factorization.factors.clone();
    let mut log_det_tt = 0.0_f64;
    for row in 0..htt_factors.len() {
        let factor = htt_factors.factor(row);
        for axis in 0..factor.nrows() {
            log_det_tt += 2.0 * factor[[axis, axis]].ln();
        }
    }
    // #1017 Phase-3: one device-resident reduced-Schur `S·v` for the WHOLE
    // evaluation — the surrogate value ladder (two-sided deflation: block-power on
    // S + inverse subspace iteration on S⁻¹ via matrix-free CG), the λ_max bracket
    // power iteration, the SLQ probes, AND the S⁻¹·probe bundle all ride this
    // single operator (uploaded / pre-factored once). Sized against the surrogate's
    // per-evaluation apply budget (probe count × shifted-CG ladder depth). The
    // device operator carries its own residency, so the CPU `SaeResidentReducedSchur`
    // frame is only staged on the CPU lane.
    let cfg_apply_budget = lane
        .as_ref()
        .map(|s| s.cfg.num_probes.saturating_mul(sys.k))
        .unwrap_or_else(|| slq_num_probes.saturating_mul(slq_lanczos_steps));
    let device_matvec =
        maybe_build_evidence_gpu_matvec(sys, ridge_t, ridge_beta, options, cfg_apply_budget)?;
    let gpu_matvec: Option<&GpuSchurMatvec> =
        options.gpu_matvec.as_ref().or(device_matvec.as_ref());
    let resident = if gpu_matvec.is_none() {
        SaeResidentReducedSchur::build(sys, &htt_factors, &backend)
    } else {
        None
    };

    // The rational ladder solves shifted SPD systems, so its operator must
    // already carry the SAME exact-A spectral classification that SLQ applies
    // inside its quadrature.  Build the low-rank Ritz correction once per
    // evaluation on the raw reduced operator, then install it on an
    // evaluation-local system clone consumed by every power/CG/value/derivative
    // apply.  Plain-SPD lanes retain the original system exactly. A lane the caller
    // admits dense classifies every eigenvalue of the raw operator in its exact-A
    // pencil, so the Ritz pass is the rational ladder's alone (#2933 F07).
    let rational_exact_a = lane.is_some()
        && !dense_reduced_schur_admitted
        && matches!(
            options.evidence_policy,
            ArrowEvidencePolicy::UnitDeflationRefusingIndefinite { .. }
        );
    // #2731 — a majorizer lane headed for the rational ladder under `UnitDeflation`
    // pins its numerically null Ritz directions at unit stiffness the same way, with the
    // floor `slq_logdet_unit_deflated` applies. A lane the caller admits dense
    // unit-deflates off the raw operator's own eigendecomposition, so it keeps the
    // original system.
    let rational_unit_deflation = match options.evidence_policy {
        ArrowEvidencePolicy::UnitDeflation { relative_floor }
            if lane.is_some() && !dense_reduced_schur_admitted =>
        {
            Some(relative_floor)
        }
        _ => None,
    };
    let mut classified_system = if let Some(relative_floor) = rational_unit_deflation {
        let raw_op = ReducedSchurOperator::new(
            sys,
            &htt_factors,
            ridge_beta,
            &backend,
            resident.as_ref(),
        )
        .with_gpu_matvec(gpu_matvec);
        let conditioning = unit_deflation_ritz_conditioning(
            sys.k,
            |direction| raw_op.apply(direction),
            relative_floor,
            slq_lanczos_steps,
            slq_seed,
        )
        .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let mut classified = sys.clone();
        classified.exact_a_reduced_conditioning = Some(conditioning);
        Some(classified)
    } else if rational_exact_a {
        if sys.exact_a_classification.is_none() {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: "rational exact-A evidence policy requires the raw B/delta/clamp \
                         classification carrier"
                    .to_string(),
            });
        }
        let raw_op = ReducedSchurOperator::new(
            sys,
            &htt_factors,
            ridge_beta,
            &backend,
            resident.as_ref(),
        )
        .with_gpu_matvec(gpu_matvec);
        let conditioning = exact_a_ritz_conditioning(
            sys.k,
            |direction| raw_op.apply(direction),
            |direction| {
                exact_a_reduced_direction_metrics(sys, &htt_factors, ridge_beta, direction)
                    .map_err(|error| error.to_string())
            },
            slq_lanczos_steps,
            slq_seed,
        )
        .map_err(|reason| ArrowSchurError::SchurFactorFailed { reason })?;
        let mut classified = sys.clone();
        classified.exact_a_reduced_conditioning = Some(conditioning);
        Some(classified)
    } else {
        None
    };

    let lane_present = lane.is_some();
    let mut dense_pencil_verdict = None;
    let log_det_schur = match lane {
        None => {
            let slq = slq_reduced_schur_log_det(
                sys,
                &htt_factors,
                ridge_beta,
                &backend,
                resident.as_ref(),
                gpu_matvec,
                options.evidence_policy,
                slq_num_probes,
                slq_lanczos_steps,
                slq_seed,
            );
            slq?.estimate
        }
        Some(state) if dense_reduced_schur_admitted => {
            if matches!(
                options.evidence_policy,
                ArrowEvidencePolicy::UnitDeflationRefusingIndefinite { .. }
            ) {
                let verdict = dense_lane_exact_a_pencil_log_det(
                    sys,
                    &htt_factors,
                    ridge_beta,
                    &backend,
                    resident.as_ref(),
                    gpu_matvec,
                    state,
                )?;
                let log_det = verdict.log_det;
                dense_pencil_verdict = Some(verdict);
                log_det
            } else {
                dense_lane_reduced_schur_log_det(
                    sys,
                    &htt_factors,
                    ridge_beta,
                    &backend,
                    resident.as_ref(),
                    gpu_matvec,
                    options.evidence_policy,
                    state,
                )?
            }
        }
        Some(state) => {
            let dim = sys.k;
            // (Re)build the frozen plan when absent or dimension-mismatched (a
            // basin mutation changed the border); otherwise reuse the frozen Q.
            let need_build = state.plan.as_ref().map_or(true, |p| p.dim != dim);
            let mut entry_evaluation = None;
            if need_build {
                let cfg = state.cfg.clone();
                // #2731 — the conditioning above reads a FIXED-STEP Lanczos, so a
                // bottom mode its Krylov space did not resolve reaches this builder
                // unpriced, and the builder's one-sided probes cannot see it. When
                // the build refuses, search the conditioned operator for a certified
                // negative mode and hand it to the same exact-A classifier: a saddle
                // is refused with the typed marker, while a clamp basin or numerical
                // null is priced and the build retried. No certified mode means the
                // refusal was about something else, and it is returned unchanged.
                // Healthy builds never pay for the search.
                let mut priced_missed_modes = 0usize;
                let derived = loop {
                    let reason = match rational_reduced_schur_plan_derived(
                        classified_system.as_ref().unwrap_or(sys),
                        &htt_factors,
                        ridge_beta,
                        &backend,
                        resident.as_ref(),
                        gpu_matvec,
                        cfg.num_probes,
                        cfg.seed,
                        cfg.rel_tol,
                        cfg.cg_rel_tol,
                        cfg.deflation_subspace_iters,
                        cfg.deflation_target_std_err_rel,
                    ) {
                        Ok(derived) => break derived,
                        Err(reason) => reason,
                    };
                    let priced = match classified_system.as_mut() {
                        Some(classified) if priced_missed_modes < dim => {
                            price_certified_bottom_mode(
                                sys,
                                classified,
                                &htt_factors,
                                ridge_beta,
                                &backend,
                                resident.as_ref(),
                                gpu_matvec,
                                cfg.rel_tol,
                                dim,
                                slq_seed,
                                options.evidence_policy,
                            )?
                        }
                        _ => false,
                    };
                    if !priced {
                        return Err(ArrowSchurError::SchurFactorFailed {
                            reason: format!(
                                "rational log-det surrogate plan build failed for reduced Schur \
                                 dim {dim}: {reason}"
                            ),
                        });
                    }
                    priced_missed_modes += 1;
                };
                state.plan = Some(derived.plan);
                entry_evaluation = Some(derived.entry_evaluation);
                // The old-dim S⁻¹·probes are meaningless against the new border.
                state.warm_inverse_probes = None;
            }
            let evidence_system = classified_system.as_ref().unwrap_or(sys);
            let plan = state
                .plan
                .as_ref()
                .expect("plan installed just above when absent");
            let want_bundle = state.request_inverse_probes;
            let want_logdet_derivative = state.request_logdet_derivative_bundle;
            // Value, its lossless shifted derivative representation, and any
            // EFS-only `(probes, S⁻¹·probes)` trace bundle are computed under one
            // borrow of the frozen plan and stashed after that borrow ends. The
            // EFS bundle uses raw probes; the outer gradient consumes only the
            // weighted shifted derivative bundle.
            let (estimate, derivative_bundle, bundle) = {
                // #1017: ONE reduced-Schur operator for the whole value ladder —
                // the frozen plan walks its shift ladder through this single
                // resident apply instead of re-capturing a `schur_matvec` closure
                // per shifted solve. When `gpu_matvec` is `Some` (Phase-3 device
                // seam, built once above) every shifted apply runs on device; when
                // `None` the byte-identical CPU `schur_matvec` lane is taken.
                let op = ReducedSchurOperator::new(
                    evidence_system,
                    &htt_factors,
                    ridge_beta,
                    &backend,
                    resident.as_ref(),
                )
                .with_gpu_matvec(gpu_matvec);
                let matvec = |v: ArrayView1<f64>| -> Array1<f64> { op.apply(v) };
                // #2576: the EXACT diag(S) is cheap on this lane and was MEASURED
                // not to help — 5189 iterations against this shared block's 5138,
                // because the elimination term carries the same firing-count
                // structure as `H_ββ` and the two very nearly cancel to a uniform
                // rescaling, which CG is invariant to.
                let precond =
                    reduced_schur_shifted_preconditioner(evidence_system, ridge_beta);
                // The derived-plan builder already certified this exact plan on
                // this exact entry operator with this exact preconditioner. Keep
                // that evaluation as the first value/derivative payload instead
                // of immediately walking the whole shifted-PCG ladder a second
                // time. Subsequent ρ values evaluate the frozen plan normally.
                let eval = match entry_evaluation.take() {
                    Some(eval) => eval,
                    None => {
                        let seed_shift = plan
                            .nodes
                            .iter()
                            .map(|(t, _)| *t)
                            .fold(f64::INFINITY, f64::min);
                        let cg_budget = plan
                            .cg_iteration_bound(&precond, seed_shift, state.cfg.cg_rel_tol)
                            .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                                reason: format!(
                                    "rational log-det surrogate has no finite conjugate-gradient \
                                     iteration bound at seed shift {seed_shift:.6e} on bracket \
                                     [{:.6e}, {:.6e}] (reduced Schur dim {dim})",
                                    plan.lambda_min, plan.lambda_max
                                ),
                            })?;
                        plan.evaluate_family_preconditioned(
                            &matvec,
                            &precond,
                            state.cfg.cg_rel_tol,
                            cg_budget,
                        )
                        .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                            reason: format!(
                                "rational log-det surrogate evaluation refused: a shifted-CG \
                                 solve broke down or did not certify within its Chebyshev \
                                 iteration bound of {cg_budget} (reduced Schur dim {dim})"
                            ),
                        })?
                    }
                };
                let estimate = eval.estimate;
                let derivative_bundle = if want_logdet_derivative {
                    Some(
                        plan.into_directional_derivative_bundle(eval)
                            .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                                reason: "rational log-det derivative bundle assembly failed"
                                    .to_string(),
                            })?,
                    )
                } else {
                    None
                };
                let bundle = if want_bundle {
                    let cg_budget = plan
                        .cg_iteration_bound(&precond, 0.0, state.cfg.cg_rel_tol)
                        .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                            reason: format!(
                                "rational surrogate inverse-probe bundle has no finite \
                                 conjugate-gradient iteration bound on bracket [{:.6e}, {:.6e}] \
                                 (reduced Schur dim {dim})",
                                plan.lambda_min, plan.lambda_max
                            ),
                        })?;
                    let (sinv, cg_report) = reduced_schur_inverse_probe_solves(
                        evidence_system,
                        &htt_factors,
                        ridge_beta,
                        &backend,
                        resident.as_ref(),
                        gpu_matvec,
                        &plan.probes,
                        state.warm_inverse_probes.as_deref(),
                        state.cfg.cg_rel_tol,
                        cg_budget,
                    )
                    .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                        reason: "rational surrogate inverse-probe bundle solve failed".to_string(),
                    })?;
                    if !cg_report.converged() {
                        log::debug!(
                            "rational surrogate inverse-probe bundle: weakest reduced-Schur CG \
                             reached relative residual {:.3e} against tolerance {:.3e} after \
                             {} of {} iterations (preconditioner {:?}); every trace contracted \
                             against this bundle inherits that error",
                            cg_report.relative_residual,
                            cg_report.tolerance,
                            cg_report.iterations,
                            cg_report.max_iterations,
                            cg_report.preconditioner,
                        );
                    }
                    Some((plan.probes.clone(), sinv))
                } else {
                    None
                };
                (estimate, derivative_bundle, bundle)
            };
            if want_logdet_derivative {
                state.logdet_derivative_bundle = derivative_bundle;
                state.request_logdet_derivative_bundle = false;
            }
            if want_bundle {
                // Keep the fresh solves as the next ρ's warm-start seed (CRN),
                // then hand the bundle to the gradient lane.
                if let Some((_, sinv)) = &bundle {
                    state.warm_inverse_probes = Some(sinv.clone());
                }
                state.inverse_probes = bundle;
                state.request_inverse_probes = false;
            }
            estimate
        }
    };
    // The reduced-Schur directions the shared exact-A classifier priced at their clamp
    // basin. The dense exact-A lane counts them off its pencil; the rational lane reads
    // them off the conditioning it accumulates (its Ritz conditioning plus any missed
    // bottom mode priced afterwards). The lane-free SLQ route classifies inside its
    // quadrature and surfaces no verdict, so it reports none rather than a zero.
    let (reduced_clamp_basin_directions, reduced_band_directions) = match dense_pencil_verdict {
        Some(verdict) => (Some(verdict.clamp_basin_directions), verdict.band_directions),
        None => (
            lane_present.then(|| {
                classified_system
                    .as_ref()
                    .and_then(|system| system.exact_a_reduced_conditioning.as_ref())
                    .map_or(0, |conditioning| conditioning.clamp_basin_directions)
            }),
            Vec::new(),
        ),
    };
    Ok((
        log_det_tt,
        log_det_schur,
        factorization,
        reduced_clamp_basin_directions,
        reduced_band_directions,
    ))
}

/// #2731 — the host bytes a caller admits `dense_lane_reduced_schur_log_det` at, for
/// a reduced Schur of dimension `k`. At most four `k × k` blocks are alive at once in
/// either phase of the route. While it materializes and eigendecomposes, those are
/// the applied operator, the eigendecomposition's working copy, its eigenvectors and
/// their returned copy. While it emits, they are the eigenvectors, the derivative
/// bundle, and the EFS probes with their inverse images. The eigendecomposition's
/// own internal storage is not bounded here, so two more blocks are reserved for it:
/// the same six-block eigensystem workspace the support lane's dense eigensystems
/// are admitted at. `None` when the byte count overflows `usize`.
///
/// This is the admission of the majorizer policies' lane. An evaluation under
/// [`ArrowEvidencePolicy::UnitDeflationRefusingIndefinite`] takes the exact-A pencil lane
/// instead, admitted at [`dense_lane_exact_a_pencil_peak_bytes`] (#2933 F07).
pub fn dense_lane_reduced_schur_peak_bytes(k: usize) -> Option<usize> {
    const DENSE_LANE_BLOCKS: usize = 6;
    k.checked_mul(k)?
        .checked_mul(std::mem::size_of::<f64>())?
        .checked_mul(DENSE_LANE_BLOCKS)
}

/// #2933 F07 — the host bytes a caller admits `dense_lane_exact_a_pencil_log_det` at, the
/// lane an evaluation under [`ArrowEvidencePolicy::UnitDeflationRefusingIndefinite`] takes
/// when admitted, for a reduced Schur of dimension `k`. The lane holds its four pencil
/// operands and the materialized operator, and frees each block once it is replaced: the
/// metric once factored, the operator once whitened, the half-whitened product once
/// whitened again. While it eigendecomposes, six `k × k` blocks own storage: the
/// substituted, clamp and lift forms, the metric factor, the whitened operator and its
/// eigenvectors. While it emits, at most five do: the pencil directions, the derivative
/// bundle, the priced inverse, and the EFS probes with their inverse images. Two more
/// blocks are reserved for the eigendecomposition's internal storage, as
/// [`dense_lane_reduced_schur_peak_bytes`] reserves them. `None` when the byte count
/// overflows `usize`.
pub fn dense_lane_exact_a_pencil_peak_bytes(k: usize) -> Option<usize> {
    const DENSE_PENCIL_LANE_BLOCKS: usize = 8;
    k.checked_mul(k)?
        .checked_mul(std::mem::size_of::<f64>())?
        .checked_mul(DENSE_PENCIL_LANE_BLOCKS)
}

/// #2731 — the lane's reduced-Schur `log|S|` when the caller's memory planner
/// admits the dense `k × k` block. `k` applies materialize the operator the
/// rational surrogate would otherwise walk by conjugate gradients, and one
/// symmetric eigendecomposition `S = V Λ Vᵀ` gives the exact value `Σ ln λ_i`.
/// Its derivative representation is exact as well, `tr(S⁻¹·D) = (1/k) Σ_i x_iᵀ D
/// x_i` with `x_i = √(k/λ_i)·v_i`, and the requested EFS pairs
/// `(√k·v_i, √k·v_i/λ_i)` average to the exact `tr(S⁻¹·M)`. No plan is frozen:
/// the value is one exact function of ρ, with no probe or quadrature error to
/// hold fixed across the search.
///
/// This lane serves the majorizer policies: an operator that is not positive definite
/// and not unit-deflated is refused with its spectrum. Under the exact-A policy the lane
/// classifies the raw operator in its pencil instead, in
/// [`dense_lane_exact_a_pencil_log_det`] (#2933 F07). Jobs 578261 and 581909
/// (`p = 2048, charts = 32`, reduced Schur dim 288, a 0.63 MiB block) refused the
/// criterion at the rational ladder's former rank ceiling 128.
///
/// Under [`ArrowEvidencePolicy::UnitDeflation`] the evidence operator is a PSD
/// majorizer, so an eigenvalue under the policy's floor, negative ones included, is
/// a numerically null direction. The lane pins it to unit stiffness exactly as
/// `factor_evidence_unit_deflated_schur` and `slq_logdet_unit_deflated` do: it adds
/// `log 1 = 0`, and the derivative bundle and the EFS pairs carry the conditioned
/// inverse `1/1` along it, the direct route's factor. Job 623388 (`7afcb5fac`,
/// `pair_chart_fit_is_certified_reml_on_a_noisy_ring`) refused the support lane's
/// spectrum `[-6.161538e-16, 2.584721e1]` (dim 24) before this.
fn dense_lane_reduced_schur_log_det<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    evidence_policy: ArrowEvidencePolicy,
    state: &mut SurrogateLaneState,
) -> Result<f64, ArrowSchurError> {
    let dim = sys.k;
    let schur = dense_lane_materialize_reduced_schur(
        sys,
        htt_factors,
        ridge_beta,
        backend,
        resident,
        gpu_matvec,
    )?;
    let (eigenvalues, eigenvectors) = match schur.eigh(Side::Lower) {
        Ok(decomposition) => decomposition,
        Err(error) => {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "the dense lane reduced Schur (dim {dim}) did not eigendecompose: {error}"
                ),
            });
        }
    };
    drop(schur);
    let eigenvalues = if let ArrowEvidencePolicy::UnitDeflation { relative_floor } =
        evidence_policy
    {
        let max_abs = eigenvalues
            .iter()
            .fold(0.0_f64, |acc, &lambda| acc.max(lambda.abs()));
        if !(relative_floor.is_finite() && relative_floor > 0.0 && max_abs > 0.0) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "the dense lane reduced Schur (dim {dim}) cannot be unit-deflated: \
                     relative floor {relative_floor:.3e}, spectral radius {max_abs:.3e}"
                ),
            });
        }
        let floor = relative_floor * max_abs * (1.0 - SPECTRAL_DEFLATION_HYSTERESIS_FRACTION);
        eigenvalues.mapv(|lambda| {
            if lambda.is_finite() && lambda >= floor {
                lambda
            } else {
                1.0
            }
        })
    } else if eigenvalues.iter().all(|&lambda| lambda > 0.0) {
        eigenvalues
    } else {
        let (lambda_min, lambda_max) = eigenvalues.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(low, high), &lambda| (low.min(lambda), high.max(lambda)),
        );
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "the dense lane reduced Schur is not positive definite, so log|S| is not \
                 defined at this iterate: spectrum [{lambda_min:.6e}, {lambda_max:.6e}] \
                 (dim {dim})"
            ),
        });
    };
    let log_det = eigenvalues.iter().map(|&lambda| lambda.ln()).sum::<f64>();
    if !log_det.is_finite() {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!("the dense lane reduced Schur log|S| is non-finite (dim {dim})"),
        });
    }
    if state.request_logdet_derivative_bundle {
        let bundle =
            RationalLogdetDerivativeBundle::from_positive_spectrum(&eigenvalues, &eigenvectors)
                .ok_or_else(|| ArrowSchurError::SchurFactorFailed {
                    reason: format!(
                        "the dense lane log|S| derivative bundle is not representable: an \
                         eigenvalue's inverse square root is non-finite (dim {dim})"
                    ),
                })?;
        state.logdet_derivative_bundle = Some(bundle);
        state.request_logdet_derivative_bundle = false;
    }
    if state.request_inverse_probes {
        let scale = (dim as f64).sqrt();
        let probes: Vec<Array1<f64>> = (0..dim)
            .map(|index| eigenvectors.column(index).mapv(|value| value * scale))
            .collect();
        let inverse_probes = probes
            .iter()
            .zip(eigenvalues.iter())
            .map(|(probe, &lambda)| probe.mapv(|value| value / lambda))
            .collect();
        state.inverse_probes = Some((probes, inverse_probes));
        state.request_inverse_probes = false;
    }
    Ok(log_det)
}

/// The dense lane's materialized reduced operator: `k` applies of the operator `sys`
/// represents, symmetrized (#2731).
fn dense_lane_materialize_reduced_schur<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
) -> Result<Array2<f64>, ArrowSchurError> {
    let dim = sys.k;
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    let mut schur = Array2::<f64>::zeros((dim, dim));
    let mut unit = Array1::<f64>::zeros(dim);
    let mut image = Array1::<f64>::zeros(dim);
    for column in 0..dim {
        unit[column] = 1.0;
        op.apply_into(&unit, &mut image);
        unit[column] = 0.0;
        schur.column_mut(column).assign(&image);
    }
    for row in 0..dim {
        for column in row + 1..dim {
            let mean = 0.5 * (schur[[row, column]] + schur[[column, row]]);
            schur[[row, column]] = mean;
            schur[[column, row]] = mean;
        }
    }
    if schur.iter().any(|value| !value.is_finite()) {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!("the dense lane reduced Schur (dim {dim}) has a non-finite entry"),
        });
    }
    Ok(schur)
}

/// What the dense exact-A lane priced: its value, the directions it priced at a clamp
/// basin, and its band.
struct DenseLanePencilVerdict {
    log_det: f64,
    clamp_basin_directions: usize,
    band_directions: Vec<ReducedBandDirection>,
}

/// #2933 F07 — the dense lane's reduced-Schur `log|S_A|` under
/// [`ArrowEvidencePolicy::UnitDeflationRefusingIndefinite`]. EVERY eigenvalue of the raw
/// reduced exact-A operator is classified in the pencil `S_A w = μ Y w`, on the band the
/// dense joint route classifies on, so no Krylov or Ritz pass decides a pin.
///
/// `Y`, the substituted form and the clamp form `E` are the Ritz forms of `(A, Φ)` on the
/// `A`-lift, [`ExactAReducedPencilOperands`]. With `Y = LLᵀ` the lane diagonalizes
/// `L⁻¹S_A L⁻ᵀ = UΛUᵀ`, so `W = L⁻ᵀU` satisfies `WᵀYW = I`. Once repeated eigenspaces
/// are resolved against the substitution ([`canonicalize_exact_a_rank_clusters`]), each
/// direction reads its edge [`exact_a_band_edge`] on its substituted stiffness
/// `s = wᵀ(Φ − B_raw)w` and its resolution `τ` from the joint operands:
///
/// * `μ > edge` is retained;
/// * `|μ| ≤ edge` is in the band, priced at the metric's own curvature `μ̃ = 1` and
///   returned as a typed [`ReducedBandDirection`];
/// * `μ < −edge` joins the negative subspace `W_N`, which is priced as a whole: the basin
///   `C = diag(μ_N) + W_NᵀEW_N` is diagonalized, a basin curvature `κ` above its floor is
///   priced, one inside it is in the band, and one below it refuses the evaluation with
///   the typed indefinite-evidence marker.
///
/// The value is `log|Y| + Σ ln μ + Σ ln κ`, the log-determinant of the priced operator
/// `S̃ = W⁻ᵀ diag(μ̃) W⁻¹`. With an empty band and no negative direction it is `log|S_A|`,
/// since `det S_A = det Y · Π μ`. The derivative bundle carries `√(k/μ̃)·w`, so
/// `(1/k) Σ x xᵀ = W diag(1/μ̃) Wᵀ = S̃⁻¹`. `W` is not orthonormal, so the EFS pairs are
/// `(√k·eᵢ, S̃⁻¹·√k·eᵢ)`. Along a band direction the value's derivative is `wᵀ dY w`, a
/// channel this crate has no `dY` for; the bundle contracts `wᵀ dS_A w` there, as a unit
/// pin does on the other lanes.
fn dense_lane_exact_a_pencil_log_det<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    state: &mut SurrogateLaneState,
) -> Result<DenseLanePencilVerdict, ArrowSchurError> {
    let dim = sys.k;
    let refusal = |reason: String| ArrowSchurError::SchurFactorFailed { reason };
    let ExactAReducedPencilOperands {
        metric,
        substituted_metric,
        clamp_metric,
        lift_gram,
        joint_dimension,
        operator_frobenius,
        metric_frobenius,
        clamp_frobenius,
    } = exact_a_reduced_pencil_operands(sys, htt_factors)?;
    let schur = dense_lane_materialize_reduced_schur(
        sys,
        htt_factors,
        ridge_beta,
        backend,
        resident,
        gpu_matvec,
    )?;
    let lower = cholesky_lower(&metric).map_err(|reason| {
        refusal(format!(
            "the dense lane exact-A pencil metric (dim {dim}) is not positive definite: {reason}"
        ))
    })?;
    drop(metric);
    let metric_log_det = (0..dim)
        .map(|axis| 2.0 * lower[[axis, axis]].ln())
        .sum::<f64>();
    let half = forward_substitution_lower_matrix(&lower, &schur);
    drop(schur);
    let mut whitened = forward_substitution_lower_matrix(&lower, half.t());
    drop(half);
    for row in 0..dim {
        for column in (row + 1)..dim {
            let mean = 0.5 * (whitened[[row, column]] + whitened[[column, row]]);
            whitened[[row, column]] = mean;
            whitened[[column, row]] = mean;
        }
    }
    let (mut curvatures, rotation) = whitened.eigh(Side::Lower).map_err(|error| {
        refusal(format!(
            "the dense lane exact-A pencil (dim {dim}) did not eigendecompose: {error}"
        ))
    })?;
    drop(whitened);
    let mut directions = Array2::<f64>::zeros((dim, dim));
    for column in 0..dim {
        directions
            .column_mut(column)
            .assign(&gam_linalg::triangular::back_substitution_lower_transpose(
                &lower,
                rotation.column(column),
            ));
    }
    drop(rotation);
    drop(lower);
    let spectral_norm = curvatures
        .iter()
        .fold(0.0_f64, |acc, &curvature| acc.max(curvature.abs()));
    canonicalize_exact_a_rank_clusters(
        &mut curvatures,
        &mut directions,
        spectral_norm,
        &|direction| Ok(substituted_metric.dot(direction)),
    )
    .map_err(|reason| refusal(format!("the dense lane exact-A pencil (dim {dim}): {reason}")))?;

    let mut priced = Array1::<f64>::ones(dim);
    let mut log_det = metric_log_det;
    let mut band_directions = Vec::new();
    let mut negative = Vec::new();
    let mut resolution_crossings = 0usize;
    for index in 0..dim {
        let curvature = curvatures[index];
        let direction = directions.column(index);
        let substituted = direction.dot(&substituted_metric.dot(&direction));
        if !(curvature.is_finite() && substituted.is_finite()) {
            return Err(refusal(format!(
                "the dense lane exact-A pencil direction {index} (dim {dim}) has curvature \
                 {curvature:e} and substituted stiffness {substituted:e}"
            )));
        }
        // A direction whose pins lower curvature more than they raise it carries no
        // substituted stiffness, so the total is clamped at zero.
        let substituted_stiffness = substituted.max(0.0);
        let resolution = exact_a_pencil_resolution(
            joint_dimension,
            direction.dot(&lift_gram.dot(&direction)),
            operator_frobenius,
            metric_frobenius,
            curvature,
        );
        if curvature.abs() > exact_a_pencil_floor() && curvature.abs() <= resolution {
            resolution_crossings += 1;
        }
        let edge = exact_a_band_edge(curvature, resolution, substituted_stiffness);
        if curvature > edge {
            priced[index] = curvature;
            log_det += curvature.ln();
        } else if curvature < -edge {
            negative.push(index);
        } else {
            band_directions.push(ReducedBandDirection {
                direction: direction.to_owned(),
                curvature,
                edge,
                resolution,
                substituted_stiffness,
                origin: ReducedBandOrigin::Pencil,
            });
        }
    }
    drop(substituted_metric);

    let mut clamp_basin_directions = 0usize;
    if !negative.is_empty() {
        let width = negative.len();
        let basis = Array2::from_shape_fn((dim, width), |(row, column)| {
            directions[[row, negative[column]]]
        });
        let mut basin = basis.t().dot(&clamp_metric.dot(&basis));
        for (position, &index) in negative.iter().enumerate() {
            basin[[position, position]] += curvatures[index];
        }
        for row in 0..width {
            for column in (row + 1)..width {
                let mean = 0.5 * (basin[[row, column]] + basin[[column, row]]);
                basin[[row, column]] = mean;
                basin[[column, row]] = mean;
            }
        }
        let (basin_curvatures, basin_rotation) = basin.eigh(Side::Lower).map_err(|error| {
            refusal(format!(
                "the dense lane exact-A basin (width {width}) did not eigendecompose: {error}"
            ))
        })?;
        let basin_directions = basis.dot(&basin_rotation);
        let mut refused: Vec<(f64, f64)> = Vec::new();
        for (position, &curvature) in basin_curvatures.iter().enumerate() {
            let direction = basin_directions.column(position);
            if !curvature.is_finite() {
                return Err(refusal(format!(
                    "the dense lane exact-A basin direction {position} (width {width}) has \
                     curvature {curvature:e}"
                )));
            }
            // `E` enters the basin at its own scale, so its norm joins the operator's in
            // the resolution, as on the dense joint route.
            let resolution = exact_a_pencil_resolution(
                joint_dimension,
                direction.dot(&lift_gram.dot(&direction)),
                operator_frobenius + clamp_frobenius,
                metric_frobenius,
                curvature,
            );
            let edge = exact_a_band_edge(curvature, resolution, 0.0);
            let index = negative[position];
            directions.column_mut(index).assign(&direction);
            if curvature < -edge {
                refused.push((curvature, edge));
            } else if curvature > edge {
                priced[index] = curvature;
                log_det += curvature.ln();
                clamp_basin_directions += 1;
            } else {
                band_directions.push(ReducedBandDirection {
                    direction: direction.to_owned(),
                    curvature,
                    edge,
                    resolution,
                    substituted_stiffness: 0.0,
                    origin: ReducedBandOrigin::Basin,
                });
            }
        }
        if let Some(&(curvature, edge)) = refused.first() {
            return Err(refusal(format!(
                "the dense lane reduced Schur {}: {} of the {width} directions of its exact-A \
                 pencil's negative subspace keep negative basin curvature once the clamp is \
                 restored (most negative {curvature:.6e} against floor {edge:.6e}, dim {dim}); \
                 the shared majorizer-metric classifier declares a genuine saddle (#2933 F07)",
                ArrowSchurError::indefinite_evidence_marker(),
                refused.len(),
            )));
        }
    }
    drop(clamp_metric);
    drop(lift_gram);
    if !log_det.is_finite() {
        return Err(refusal(format!(
            "the dense lane exact-A log|S_A| is non-finite (dim {dim})"
        )));
    }
    if resolution_crossings > 0 {
        log::debug!(
            "[dense lane exact-A] numerical resolution limit: {resolution_crossings} of {dim} \
             pencil directions clear √ε but not their own numerical resolution; the band \
             prices them at the metric's curvature"
        );
    }
    if state.request_logdet_derivative_bundle {
        let bundle = RationalLogdetDerivativeBundle::from_positive_spectrum(&priced, &directions)
            .ok_or_else(|| {
                refusal(format!(
                    "the dense lane exact-A log|S_A| derivative bundle is not representable: a \
                     priced curvature's inverse square root is non-finite (dim {dim})"
                ))
            })?;
        state.logdet_derivative_band_directions = band_directions.clone();
        state.logdet_derivative_bundle = Some(bundle);
        state.request_logdet_derivative_bundle = false;
    }
    if state.request_inverse_probes {
        let scale = (dim as f64).sqrt();
        let inverse = (&directions * &priced.mapv(f64::recip)).dot(&directions.t());
        let probes: Vec<Array1<f64>> = (0..dim)
            .map(|index| {
                let mut probe = Array1::<f64>::zeros(dim);
                probe[index] = scale;
                probe
            })
            .collect();
        let inverse_probes = (0..dim)
            .map(|index| inverse.column(index).mapv(|value| value * scale))
            .collect();
        state.inverse_probes = Some((probes, inverse_probes));
        state.request_inverse_probes = false;
    }
    Ok(DenseLanePencilVerdict {
        log_det,
        clamp_basin_directions,
        band_directions,
    })
}

/// #2731 — find and price ONE bottom mode of the conditioned rational exact-A
/// operator that its fixed-step conditioning did not resolve.
///
/// Runs the folded Lanczos of `reduced_schur_negative_curvature` on `classified`,
/// the operator the plan builder actually solved, for at most `max_steps` (the
/// builder's own seed-solve budget, within which the ladder broke down). A
/// certified mode is orthogonalized against the directions already priced,
/// re-measured on the RAW exact-A operator, and handed to the shared classifier
/// with its majorizer and clamp metrics:
///
/// * `Saddle` → the typed `indefinite_evidence_marker` refusal, which the outer
///   search maps to an infeasible probe;
/// * `ClampBasin` / `NumericalNull` → appended to the conditioning at its price,
///   returning `Ok(true)` so the caller retries the build;
/// * no certified negative mode, or no positive spectral bound to fold about →
///   `Ok(false)`. Declining to measure is not a verdict, so the caller returns the
///   builder's own refusal unchanged.
fn price_certified_bottom_mode<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    classified: &mut ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    rel_tol: f64,
    max_steps: usize,
    seed: u64,
    evidence_policy: ArrowEvidencePolicy,
) -> Result<bool, ArrowSchurError> {
    let Some(lambda_max) = reduced_schur_lambda_max(
        classified,
        htt_factors,
        ridge_beta,
        backend,
        resident,
        gpu_matvec,
        rel_tol,
        seed,
    ) else {
        return Ok(false);
    };
    let Some(mode) = reduced_schur_negative_curvature(
        classified,
        htt_factors,
        ridge_beta,
        backend,
        resident,
        gpu_matvec,
        lambda_max,
        max_steps,
        seed,
    ) else {
        return Ok(false);
    };
    let (mut directions, mut shifts, mut clamp_basin_directions) = classified
        .exact_a_reduced_conditioning
        .as_ref()
        .map_or_else(
            || (Vec::new(), Vec::new(), 0usize),
            |conditioning| {
                (
                    conditioning.directions.to_vec(),
                    conditioning.shifts.to_vec(),
                    conditioning.clamp_basin_directions,
                )
            },
        );
    // Ritz vectors of one symmetric operator are orthogonal, but this mode comes
    // from a different Krylov run, so remove what the priced span already carries
    // before its price is added as a rank-1 correction.
    let mut direction = mode.border;
    for _ in 0..2 {
        for priced in &directions {
            let overlap = priced.dot(&direction);
            direction.scaled_add(-overlap, priced);
        }
    }
    let norm = direction.dot(&direction).sqrt();
    if !(norm.is_finite() && norm > f64::EPSILON.sqrt()) {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "rational exact-A conditioning: the certified bottom mode (curvature {:.6e}) \
                 lies in the span already priced (residual norm {norm:.3e}), so the carrier \
                 and the operator disagree",
                mode.curvature
            ),
        });
    }
    direction.mapv_inplace(|value| value / norm);
    let raw = direction.dot(
        &ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
            .with_gpu_matvec(gpu_matvec)
            .apply(direction.view()),
    );
    let priced = if let ArrowEvidencePolicy::UnitDeflation { .. } = evidence_policy {
        // #2731 — under `UnitDeflation` the evidence operator is a PSD majorizer, so a
        // certified negative mode is a numerically null direction: unit stiffness, as
        // `unit_deflation_ritz_conditioning` pins the modes it resolves.
        log::debug!(
            "[rational unit deflation] pinned a bottom mode its fixed-step conditioning \
             missed at unit stiffness: raw curvature {raw:.6e} ({} directions pinned)",
            directions.len() + 1
        );
        1.0
    } else {
        let (majorizer, clamp) =
            exact_a_reduced_direction_metrics(sys, htt_factors, ridge_beta, direction.view())?;
        let priced = match classify_exact_a_direction(
            raw,
            sys.k,
            lambda_max.max(raw.abs()),
            majorizer,
            clamp,
        ) {
            ExactADirectionClassification::NumericalNull => 1.0,
            ExactADirectionClassification::ClampBasin { curvature } => {
                clamp_basin_directions += 1;
                curvature
            }
            ExactADirectionClassification::Saddle { curvature, basin } => {
                return Err(ArrowSchurError::SchurFactorFailed {
                    reason: format!(
                        "matrix-free reduced-Schur {}: the rational ladder's bottom mode, missed \
                         by its fixed-step conditioning, has raw exact-A curvature \
                         {curvature:.6e} and clamp basin {basin:.6e}; the shared \
                         majorizer-metric classifier declares a genuine saddle (#2731/#2515)",
                        ArrowSchurError::indefinite_evidence_marker(),
                    ),
                });
            }
            ExactADirectionClassification::ResolvedPositive { curvature } => {
                return Err(ArrowSchurError::SchurFactorFailed {
                    reason: format!(
                        "rational exact-A conditioning: the conditioned operator certifies \
                         curvature {:.6e} along a direction whose raw exact-A curvature \
                         {curvature:.6e} resolves positive, so the carrier and the operator \
                         disagree",
                        mode.curvature
                    ),
                });
            }
        };
        log::debug!(
            "[rational exact-A] priced a bottom mode its fixed-step conditioning missed: raw \
             curvature {raw:.6e}, majorizer {majorizer:.6e}, clamp {clamp:.6e}, priced \
             {priced:.6e} ({} directions priced)",
            directions.len() + 1
        );
        priced
    };
    directions.push(direction);
    shifts.push(priced - raw);
    classified.exact_a_reduced_conditioning = Some(ExactAReducedRitzConditioning {
        directions: directions.into(),
        shifts: shifts.into(),
        clamp_basin_directions,
    });
    Ok(true)
}

/// Certified upper bracket on the largest eigenvalue of the reduced Schur
/// complement, for sizing the rational quadrature window.
///
/// Returns `θ + r`: the top Ritz value of an adaptive, fully reorthogonalized
/// Lanczos solve on `S`, plus its sharp Ritz residual bound `r = β_j |e_jᵀ y|`.
/// For a symmetric operator `θ <= λ_max <= θ + r`, so this is an upper bound on
/// `λ_max` rather than an estimate from below. The quadrature window keeps
/// `λ_max`'s tail truncation at `λ_max / t_hi`, and an estimate from below
/// could leave that tail over its tolerance by the estimate's shortfall.
///
/// The solve stops at the first step whose relative residual clears `rel_tol`,
/// the accuracy the caller sizes its quadrature for. The work bound is the
/// algebraic span `k`: a reorthogonalized Krylov recurrence has exhausted `S`
/// after `k` steps. The tridiagonal certificate is checked every step, because
/// its `O(j²)` cost is negligible beside one `S·v`. The former fixed
/// power-iteration count answered neither question.
///
/// `None` when `k == 0`, when the dominant-magnitude Ritz value is not positive
/// (an operator whose most negative eigenvalue dominates has no positive top
/// bracket to report), or when the solve does not certify.
pub(crate) fn reduced_schur_lambda_max<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    rel_tol: f64,
    seed: u64,
) -> Option<f64> {
    let k = sys.k;
    if k == 0 || !(rel_tol.is_finite() && rel_tol > 0.0) {
        return None;
    }
    // Deterministic Rademacher start (same stream discipline as the surrogate
    // probes): a ±1 vector never lands orthogonal to the top eigenspace.
    let mut start = vec![0.0_f64; k];
    {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut bits: u64 = 0;
        let mut remaining: u32 = 0;
        for value in start.iter_mut() {
            if remaining == 0 {
                bits = gam_linalg::utils::splitmix64(&mut state);
                remaining = 64;
            }
            *value = if bits & 1 == 1 { 1.0 } else { -1.0 };
            bits >>= 1;
            remaining -= 1;
        }
    }
    // One resident operator reused across every Lanczos apply — device seam
    // threaded so the bracket rides the SAME resident `S·v` the ladder/probes use.
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    let options = gam_linalg::lanczos::SymmetricExtremeLanczosOptions {
        target_rank: 1,
        max_steps: k,
        check_every: 1,
        relative_residual_tol: rel_tol,
        breakdown_tol: 0.0,
    };
    let mut work = Array1::<f64>::zeros(k);
    let pairs = gam_linalg::lanczos::symmetric_extreme_lanczos_eigenpairs(
        k,
        &start,
        options,
        |x: &[f64], out: &mut [f64]| {
            let xv = Array1::from_iter(x.iter().copied());
            op.apply_into(&xv, &mut work);
            for (slot, &sv) in out.iter_mut().zip(work.iter()) {
                *slot = sv;
            }
            Ok(())
        },
    )
    .ok()?;
    let ritz = *pairs.eigenvalues.first()?;
    let bound = *pairs.residual_bounds.first()?;
    let upper = ritz + bound;
    (ritz.is_finite() && bound.is_finite() && ritz > 0.0 && upper.is_finite()).then_some(upper)
}

/// A measured direction of negative curvature of the reduced Schur complement,
/// carried into the FULL arrow coordinates.
///
/// `curvature` is the Rayleigh quotient `vᵀSv` re-measured with one extra
/// `S·v` apply, not the Ritz value the eigensolver reported — the Ritz value
/// is an estimate from a Krylov space, and a certificate of indefiniteness must
/// be an evaluation of the operator itself. `border` is the unit mode `v` in
/// the reduced (border) coordinates and `eliminated` is its exact lift
/// `L(v)` through the arrow elimination, so `(eliminated, border)` is a
/// displacement of the full system whose curvature is exactly `curvature`.
#[derive(Debug, Clone)]
pub(crate) struct ReducedSchurNegativeCurvature {
    /// `vᵀSv < 0`, measured by an apply rather than reported by the eigensolver.
    pub curvature: f64,
    /// The algebraically smallest Ritz value the shifted solve certified.
    pub ritz_eigenvalue: f64,
    /// The shift `σ ≥ λ_max` the spectral fold used.
    pub shift: f64,
    /// The unit mode in reduced/border coordinates.
    pub border: Array1<f64>,
    /// `L(v)`: the same mode in the eliminated blocks' coordinates.
    pub eliminated: Array1<f64>,
}

/// The reduced Schur's algebraically most-negative eigenpair, matrix-free, and
/// the full-space displacement it lifts to — `None` when the operator resolves
/// no negative direction.
///
/// # Why a shift rather than plain Lanczos
///
/// [`gam_linalg::lanczos::symmetric_extreme_lanczos_eigenpairs`] certifies
/// extreme-MAGNITUDE eigenpairs. At a saddle of a penalized fit `λ_max` is the
/// data curvature and `λ_min` is a small negative number, so the largest
/// magnitude is the wrong end and the mode that matters is invisible to it.
/// Running the same solver on `σI − S` fixes that exactly: the spectrum folds
/// to `σ − λ_j ≥ 0`, its largest element is `σ − λ_min`, and largest-magnitude
/// is now the end we want. The fold is an exact similarity on the eigenvectors
/// — it changes which eigenvalue is extreme and nothing else — and `σ` is the
/// `λ_max` the surrogate's spectral bracket already estimates
/// (`reduced_schur_lambda_max`), so no new spectral information is needed.
///
/// # Why this is a statement about the ITERATE
///
/// The lift `L` satisfies `[L(v); v]ᵀ H [L(v); v] = vᵀ S v` exactly (see
/// `arrow_lift_border_direction`). So a negative `curvature` here is not a
/// property of the reduced surrogate that might vanish in the full problem: it
/// is negative curvature of the fit's own objective at this point, and a fit
/// reporting convergence there has converged to something that is not a local
/// minimum.
pub(crate) fn reduced_schur_negative_curvature<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    lambda_max: f64,
    max_steps: usize,
    seed: u64,
) -> Option<ReducedSchurNegativeCurvature> {
    let k = sys.k;
    if k == 0 || !(lambda_max.is_finite() && lambda_max > 0.0) {
        return None;
    }
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    // Deterministic Rademacher start, the same stream discipline the surrogate
    // probes and `reduced_schur_lambda_max` use: reproducible across runs and
    // never orthogonal to the sought eigenspace by construction.
    let mut start = vec![0.0_f64; k];
    {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut bits: u64 = 0;
        let mut remaining: u32 = 0;
        for value in start.iter_mut() {
            if remaining == 0 {
                bits = gam_linalg::utils::splitmix64(&mut state);
                remaining = 64;
            }
            *value = if bits & 1 == 1 { 1.0 } else { -1.0 };
            bits >>= 1;
            remaining -= 1;
        }
    }
    // `σ` strictly above `λ_max` so the folded operator is positive semidefinite
    // even when the power-iteration estimate sits a rounding below the true top.
    let shift = lambda_max * (1.0 + 8.0 * f64::EPSILON.sqrt());
    let options = gam_linalg::lanczos::SymmetricExtremeLanczosOptions {
        target_rank: 1,
        max_steps: max_steps.clamp(1, k),
        check_every: 4,
        relative_residual_tol: f64::EPSILON.sqrt(),
        breakdown_tol: 0.0,
    };
    let mut work = Array1::<f64>::zeros(k);
    let pairs = gam_linalg::lanczos::symmetric_extreme_lanczos_eigenpairs(
        k,
        &start,
        options,
        |x: &[f64], out: &mut [f64]| {
            let xv = Array1::from_iter(x.iter().copied());
            op.apply_into(&xv, &mut work);
            for (slot, (&xi, &sv)) in out.iter_mut().zip(x.iter().zip(work.iter())) {
                *slot = shift * xi - sv;
            }
            Ok(())
        },
    )
    .ok()?;
    // Largest folded eigenvalue ⇒ smallest eigenvalue of `S`.
    let (best, &folded) = pairs
        .eigenvalues
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    let ritz_eigenvalue = shift - folded;
    let mode = pairs.eigenvectors.column(best).to_owned();
    let norm = mode.dot(&mode).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return None;
    }
    let border = mode / norm;
    // The certificate: an APPLY of the operator, not the eigensolver's estimate.
    let curvature = border.dot(&op.apply_owned(&border));
    if !(curvature.is_finite() && curvature < 0.0) {
        return None;
    }
    let eliminated = arrow_lift_border_direction(sys, htt_factors, border.view(), backend);
    if eliminated.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(ReducedSchurNegativeCurvature {
        curvature,
        ritz_eigenvalue,
        shift,
        border,
        eliminated,
    })
}

/// Matrix-free reduced-Schur log-determinant `log|S|` via the #2080 fixed
/// rational surrogate ([`RationalLogdetPlan`]) on the exact `schur_matvec`
/// apply — the desync-safe companion to `slq_reduced_schur_log_det`. **The
/// dense `k×k` `S` is NEVER formed.**
///
/// Returns the built plan and its evaluation so the caller can (a) read
/// `eval.estimate` = the surrogate value `L̃ ≈ log|S|` (with `eval.std_err` the
/// honest Hutchinson error bar), and (b) later contract the SAME shifted-solve
/// bundle against any per-ρ-coordinate Schur-derivative operator `∂S` via
/// `RationalLogdetPlan::directional_derivative`. Because both the value and that
/// derivative are the exact value / gradient of the ONE deterministic function
/// `L̃(ρ)` (fixed probes, fixed quadrature), the outer optimiser descends a
/// function whose gradient is its own — the objective↔gradient desync class the
/// bare SLQ value re-opened (a stochastic value paired with the analytic exact
/// gradient) is closed by construction, not by tolerance tuning.
///
/// The spectral bracket is estimated matrix-free: `λ_max` as a certified Lanczos
/// upper bracket (`reduced_schur_lambda_max`), `λ_min` from the deflation-floor
/// convention `SPECTRAL_DEFLATION_REL_FLOOR·λ_max` (the operative lower bound of
/// the unit-deflated spectrum). Every shifted solve's iteration budget is the
/// plan's Chebyshev bound (`RationalLogdetPlan::cg_iteration_bound`), not a
/// caller constant. Deterministic for a fixed
/// `(sys, htt_factors, ρ_β, resident, num_probes, seed, rel_tol, cg_rel_tol)`.
///
/// `None` when `k == 0`, the bracket estimate is degenerate, the plan cannot be
/// built, or a shifted CG solve breaks down on a non-finite operator.
pub fn rational_reduced_schur_log_det<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    num_probes: usize,
    seed: u64,
    rel_tol: f64,
    cg_rel_tol: f64,
) -> Option<(RationalLogdetPlan, RationalLogdetEval)> {
    let k = sys.k;
    if k == 0 {
        return None;
    }
    let lambda_max = reduced_schur_lambda_max(
        sys,
        htt_factors,
        ridge_beta,
        backend,
        resident,
        gpu_matvec,
        rel_tol,
        seed,
    )?;
    // λ_min from the deflation floor: after unit-deflation the operative spectrum
    // is bounded below by `SPECTRAL_DEFLATION_REL_FLOOR·λ_max` (or 1.0), so this
    // is a sound lower bracket for the quadrature window sizing. The window is
    // padded two decades below `λ_min` inside `RationalLogdetPlan::build`, so a
    // conservative (too-small) floor only widens the resolved range, never biases
    // the estimate. The bracket certifies `λ_max > 0`, so the floor is positive
    // unless it underflows, and then the plan refuses the bracket rather than
    // sizing a window from a picked subnormal (#2469).
    let lambda_min = SPECTRAL_DEFLATION_REL_FLOOR * lambda_max;
    let plan = RationalLogdetPlan::build(k, num_probes, seed, lambda_min, lambda_max, rel_tol)?;
    // One resident operator; the plan's shift ladder reuses it across every
    // shifted solve. The probes fan across rayon workers (in `evaluate`), and
    // `schur_matvec`'s own row parallelism is guarded off inside a worker, so
    // there is no nested oversubscription.
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    let matvec = |v: ArrayView1<f64>| -> Array1<f64> { op.apply(v) };
    // #2576: the exact diag(S) is reachable from `resident` and measured NOT to
    // reduce iterations — see the refutation note at the surrogate-core call site.
    let precond = reduced_schur_shifted_preconditioner(sys, ridge_beta);
    let seed_shift = plan
        .nodes
        .iter()
        .map(|(t, _)| *t)
        .fold(f64::INFINITY, f64::min);
    let cg_budget = plan.cg_iteration_bound(&precond, seed_shift, cg_rel_tol)?;
    let eval = plan.evaluate_family_preconditioned(&matvec, &precond, cg_rel_tol, cg_budget)?;
    Some((plan, eval))
}

/// Build the FROZEN #2080 surrogate plan for one outer solve, with the Hutch++
/// deflation rank DERIVED from a pilot evaluation — the build-once companion to
/// per-ρ [`RationalLogdetPlan::evaluate`]. Returns the plan (probes +
/// quadrature + frozen Hutch++ `Q`) together with the certified evaluation that
/// selected its rank. The entry evaluation is the first criterion value and
/// derivative payload: discarding it and immediately evaluating the same plan,
/// operator, and preconditioner would repeat the whole shifted-PCG ladder. Later
/// ρ values evaluate the frozen plan normally, so rank derivation remains a
/// once-per-outer-solve cost.
///
/// Derived rank (the #2080 lead ruling): a rank-0 pilot fixes the log-det scale,
/// the target bar is `deflation_target_std_err_rel · (|log|S|_pilot| + 1)` — one
/// order under the smallest tolerance the criterion feeds (the caller passes
/// `0.1 · STALL_REL_TOL`; `log|S|` is the criterion's dominant term at wide `k`
/// so `|log|S||+1` is the right objective scale to `O(1)` and the `0.1` margin
/// absorbs the loss/Occam remainder). The peel rank grows on a doubling schedule
/// until the Hutchinson error bar clears the target. The ladder's ceiling is the
/// operator's own dimension, lowered only where the host cannot hold the plan's
/// storage at that rank (#2731). It is not permission to return an
/// under-certified estimate: exhausting it before the bar clears returns `Err`
/// and the caller surfaces a typed evidence failure. A pilot already under
/// target returns the bare-Hutchinson plan. Deterministic for fixed inputs (`Q`
/// and probes are seed-derived). The
/// returned plan's `Q` is FROZEN, so
/// `RationalLogdetPlan::directional_derivative` on its evaluations is the exact
/// surrogate gradient.
pub(crate) struct DerivedRationalLogdetPlan {
    /// Frozen statistical plan selected at the entry operator.
    pub plan: RationalLogdetPlan,
    /// Certified value and shifted solves already computed while selecting the
    /// plan, consumed as the entry value and derivative payload.
    pub entry_evaluation: RationalLogdetEval,
}

pub(crate) fn rational_reduced_schur_plan_derived<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    num_probes: usize,
    seed: u64,
    rel_tol: f64,
    cg_rel_tol: f64,
    deflation_subspace_iters: usize,
    deflation_target_std_err_rel: f64,
) -> Result<DerivedRationalLogdetPlan, String> {
    let k = sys.k;
    if k == 0
        || !(cg_rel_tol.is_finite() && cg_rel_tol > 0.0 && cg_rel_tol < 1.0)
        // `+∞` is admissible: it asks for no value bar, so the pilot is the plan.
        || !(deflation_target_std_err_rel >= 0.0)
    {
        return Err(format!(
            "inadmissible surrogate request: reduced Schur dim {k}, cg_rel_tol {cg_rel_tol:.3e} \
             (needs 0 < tol < 1), deflation target {deflation_target_std_err_rel:.3e} (needs \
             non-negative; +∞ asks for no deflation)"
        ));
    }
    let lambda_max = reduced_schur_lambda_max(
        sys,
        htt_factors,
        ridge_beta,
        backend,
        resident,
        gpu_matvec,
        rel_tol,
        seed,
    )
    .ok_or_else(|| {
        format!(
            "spectral bracket unavailable: the power iteration produced no finite λ_max for \
             reduced Schur dim {k} within its {k}-step span at relative residual {rel_tol:.3e}"
        )
    })?;
    let lambda_min = SPECTRAL_DEFLATION_REL_FLOOR * lambda_max;
    let base_plan = RationalLogdetPlan::build(
        k, num_probes, seed, lambda_min, lambda_max, rel_tol,
    )
    .ok_or_else(|| {
        format!(
            "quadrature plan unbuildable on bracket [{lambda_min:.6e}, {lambda_max:.6e}] at \
             rel_tol {rel_tol:.3e} with {num_probes} probes (reduced Schur dim {k})"
        )
    })?;
    // One resident operator across the pilot, every deflation re-solve, and the
    // subspace-iteration `with_two_sided_deflation_preconditioned` applies — the whole rank-derivation
    // ladder (the two-sided deflation: block-power on S + inverse subspace
    // iteration on S⁻¹) reuses the same staged residency / device `S·v`.
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    let matvec = |v: ArrayView1<f64>| -> Array1<f64> { op.apply(v) };
    // The SAME shared-block diagonal every evaluation of this plan will use, so
    // the rank-derivation ladder is not measuring a differently-conditioned
    // iteration from the one production runs. (#2576: the exact diag(S) was
    // measured here and does not help — see the surrogate-core call site.)
    let precond = reduced_schur_shifted_preconditioner(sys, ridge_beta);
    // Rank-0 pilot: fixes the |log|S|| scale and is the answer outright when no
    // deflation is requested or the bare bar already clears the target.
    // `log|S|` exists only for a positive-definite `S`, and every shifted solve
    // below is a conjugate-gradient recurrence that assumes it. Nothing checked
    // that assumption: the bracket's LOWER end is not measured, it is set to a
    // fixed fraction of the estimated `lambda_max`, so an operator whose
    // spectrum reaches below zero was planned for as if it did not, and the
    // first thing to notice was a CG breakdown tens of iterations in, reported
    // as "no finite solution". Measured on gam#2731: `pᵀ(A+σI)p = -1.50e10` at
    // seed shift `4.2e-15`, on a fit that had already converged.
    //
    // A quadratic form is a ONE-SIDED certificate: `vᵀ(S + t_lo·I)v <= 0` proves
    // the operator is indefinite on this bracket, while a positive value over
    // finitely many probes proves nothing. So this refuses when it fires and is
    // silent otherwise — the breakdown path still catches what it misses, and
    // now names itself. The probes are the plan's own, so this costs one extra
    // operator application each and introduces no new randomness.
    let seed_shift = base_plan
        .nodes
        .iter()
        .map(|(t, _)| *t)
        .filter(|t| t.is_finite())
        .fold(f64::INFINITY, f64::min);
    if seed_shift.is_finite() {
        for (index, probe) in base_plan.probes.iter().enumerate() {
            let norm_sq = probe.dot(probe);
            if !(norm_sq > 0.0) {
                continue;
            }
            let mut shifted = matvec(probe.view());
            shifted.scaled_add(seed_shift, probe);
            let form = probe.dot(&shifted);
            if !(form.is_finite() && form > 0.0) {
                // #2731 — a probe proves indefiniteness but names no direction,
                // and a direction is the only thing an escape can use. The
                // spectrum's own most-negative mode is a shifted Lanczos away
                // (the shift is the `λ_max` this plan already estimated), and
                // the arrow elimination lifts it into a full-space
                // displacement, so the refusal reports what descends rather
                // than only that something does.
                let escape = reduced_schur_negative_curvature(
                    sys,
                    htt_factors,
                    ridge_beta,
                    backend,
                    resident,
                    gpu_matvec,
                    lambda_max,
                    k,
                    seed,
                )
                .map(|found| {
                    format!(
                        " The spectrum's most-negative direction is vᵀSv = {:.6e} (Ritz {:.6e}                          under the fold σ = {:.6e}); the arrow elimination lifts it to a                          full-space displacement of {} eliminated coordinates, whose curvature                          is that same number by the Schur identity — so the descent direction                          is available, not merely implied.",
                        found.curvature,
                        found.ritz_eigenvalue,
                        found.shift,
                        found.eliminated.len(),
                    )
                })
                .unwrap_or_else(|| {
                    " The shifted Lanczos did not certify a negative eigenpair within its                      step budget, so the probe above is the whole of the evidence."
                        .to_string()
                });
                return Err(format!(
                    "the reduced Schur is not positive definite on this bracket, so log|S| is \
                     not defined at this iterate: probe {index} gives \
                     vᵀ(S + {seed_shift:.6e}·I)v = {form:.6e} with ‖v‖² = {norm_sq:.6e} \
                     (reduced Schur dim {k}, bracket [{lambda_min:.6e}, {lambda_max:.6e}] whose \
                     lower end is SPECTRAL_DEFLATION_REL_FLOOR × λ_max, not a measured \
                     eigenvalue). A converged fit reaching here has converged to a point with \
                     negative curvature in the reduced Schur, which is a statement about the \
                     iterate, not about the surrogate.{escape}"
                ));
            }
        }
    }
    let cg_budget = base_plan
        .cg_iteration_bound(&precond, seed_shift, cg_rel_tol)
        .ok_or_else(|| {
            format!(
                "no finite conjugate-gradient iteration bound at seed shift {seed_shift:.6e} on \
                 the bracket [{lambda_min:.6e}, {lambda_max:.6e}] at cg_rel_tol \
                 {cg_rel_tol:.3e} (reduced Schur dim {k})"
            )
        })?;
    let pilot = base_plan
        .evaluate_family_preconditioned(&matvec, &precond, cg_rel_tol, cg_budget)
        .ok_or_else(|| {
            format!(
                "rank-0 pilot solve broke down: the shifted-CG family did not return a finite \
                 solution on the bracket [{lambda_min:.6e}, {lambda_max:.6e}] at cg_rel_tol \
                 {cg_rel_tol:.3e} within its Chebyshev iteration bound of {cg_budget} (reduced \
                 Schur dim {k}). The seed system's own budget is min(bound, dim) = {} \
                 iterations. The \
                 one-sided definiteness probe above did not fire, so this is either an \
                 indefiniteness those probes missed or a genuine loss of accuracy; the \
                 `[rational-logdet] shifted-CG seed breakdown` line says which.",
                cg_budget.min(k.max(1))
            )
        })?;
    let target = deflation_target_std_err_rel * (pilot.estimate.abs() + 1.0);
    if pilot.std_err <= target {
        return Ok(DerivedRationalLogdetPlan {
            plan: base_plan,
            entry_evaluation: pilot,
        });
    }
    let pilot_std_err = pilot.std_err;
    // Grow from the smallest nonzero peel rank (doubling ⇒ log-many re-solves)
    // until the bar clears. Reaching the ceiling with an over-target bar refuses
    // the surrogate rather than silently weakening the requested
    // statistical-accuracy contract.
    //
    // #2731 — the ceiling is the operator's own dimension `k`. A basis spanning
    // the whole operator projects every probe to its rounding, so the value is
    // the deterministic rational log-det, and an interior-variance operator that
    // no low rank deflates reaches its bar there instead of refusing. Job 578261
    // (`p = 2048, charts = 32`, reduced Schur dim 288) refused at the former
    // literal ceiling 128 after removing 2.3 % of the pilot variance against a
    // 1e-9 relative bar. Only memory lowers the ceiling: a rank-`r` plan freezes
    // `r` basis columns and keeps one shifted solve per column per quadrature
    // node, `(nodes + 1)·r·k` doubles, so the host admits the ranks whose storage
    // its single-materialization cap holds.
    let rank_storage_bytes = (base_plan.nodes.len() + 1)
        .saturating_mul(k)
        .saturating_mul(std::mem::size_of::<f64>());
    let admitted_rank = gam_runtime::resource::ResourcePolicy::default_library()
        .max_single_materialization_bytes
        / rank_storage_bytes;
    let cap = k.min(admitted_rank);
    // Every number the caller needs to decide whether to relax the target or take
    // the estimate as it stands.
    let ceiling_refusal = |std_err: f64, estimate: f64| {
        format!(
            "deflation reached its rank ceiling {cap} (reduced Schur dim {k}, memory admits \
             rank {admitted_rank}) with the Hutchinson bar still over target: std_err \
             {std_err:.6e} against target {target:.6e} (= {deflation_target_std_err_rel:.3e} × \
             (|estimate| + 1)), estimate {estimate:.6e}; the rank-0 pilot's bar was \
             {pilot_std_err:.6e}, so deflation removed {:.1}% of the pilot variance and \
             needed {:.1}%",
            100.0 * (1.0 - std_err / pilot_std_err),
            100.0 * (1.0 - target / pilot_std_err),
        )
    };
    if cap == 0 {
        return Err(ceiling_refusal(pilot_std_err, pilot.estimate));
    }
    let mut rank = 1usize;
    // Basis iteration only steers Q for variance reduction. Derive its looser
    // true-residual tolerance from the evaluation solve's tolerance instead of
    // carrying an unrelated fixed knob: √tol is strictly looser while still
    // converging as the bottom-tail builder now requires.
    let basis_cg_rel_tol = cg_rel_tol.sqrt();
    let basis_cg_budget = base_plan
        .cg_iteration_bound(&precond, 0.0, basis_cg_rel_tol)
        .ok_or_else(|| {
            format!(
                "no finite conjugate-gradient iteration bound for the inverse-iteration basis on \
                 the bracket [{lambda_min:.6e}, {lambda_max:.6e}] at cg_rel_tol \
                 {basis_cg_rel_tol:.3e} (reduced Schur dim {k})"
            )
        })?;
    loop {
        let r = rank.min(cap);
        // Split the peel budget across BOTH spectral tails at equal total rank:
        // the Hutchinson bar rides on ‖offdiag(P log(S/c) P)‖_F, whose mass sits
        // symmetrically on the λ_max AND λ_min tails (|log(λ/c)| peaks equally at
        // both ends of the bracket since c is its geometric midpoint), so top-only
        // deflation stalls at ~½ the removable variance
        // (`two_sided_deflation_drops_wide_kappa_std_err_below_two_percent`).
        // The bottom-tail basis comes from inverse iteration — CG on the UNSHIFTED
        // operator at full κ — so it gets its own LOOSE budget, not the
        // evaluation-grade `cg_rel_tol`: an approximate bottom `Q` only relaxes
        // the variance reduction, never biases the value (the split is exact for
        // any orthonormal `Q`), while an evaluation-grade solve there would burn
        // √κ-scale iterations per basis column for no accuracy in return.
        let plan = base_plan.clone().with_two_sided_deflation_preconditioned(
            &matvec,
            &precond,
            r.div_ceil(2),
            r / 2,
            deflation_subspace_iters,
            seed,
            (basis_cg_rel_tol, basis_cg_budget),
        )
        .ok_or_else(|| {
            format!(
                "two-sided deflation basis unbuildable at rank {r} (top {}, bottom {}) after \
                 {deflation_subspace_iters} subspace iterations on reduced Schur dim {k}",
                r.div_ceil(2),
                r / 2
            )
        })?;
        let eval = plan
            .evaluate_family_preconditioned(&matvec, &precond, cg_rel_tol, cg_budget)
            .ok_or_else(|| {
                format!(
                    "deflated solve broke down at rank {r}: the shifted-CG family did not return \
                     a finite solution at cg_rel_tol {cg_rel_tol:.3e} (reduced Schur dim {k})"
                )
            })?;
        if eval.std_err <= target {
            return Ok(DerivedRationalLogdetPlan {
                plan,
                entry_evaluation: eval,
            });
        }
        if r >= cap {
            // The resource ceiling, reached with an over-target bar. This is a
            // deliberate refusal rather than a silent weakening of the accuracy
            // contract — but a refusal that names only its dimension cannot be
            // acted on, and this one aborts a fit that has already converged.
            return Err(ceiling_refusal(eval.std_err, eval.estimate));
        }
        rank = rank.saturating_mul(2);
    }
}

/// Convergence certificate for one matrix-free reduced-Schur CG solve.
///
/// The evidence lane's `S⁻¹`-apply used to return its iterate with no way to
/// tell a converged solve from one truncated at `max_iters`: a stagnating CG
/// handed back an arbitrarily-wrong `S⁻¹b`, and every downstream trace /
/// log-determinant estimate inherited that error SILENTLY (#2576 — a 4096-cap
/// truncation invisible behind six minutes of no log output). The solve now
/// carries what it achieved so consumers can refuse, escalate, or report
/// instead of re-deriving it from nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducedSchurCgReport {
    /// CG iterations actually taken.
    pub iterations: usize,
    /// Iteration cap the solve ran under.
    pub max_iterations: usize,
    /// `‖b − S y‖ / ‖b‖` at the returned iterate.
    pub relative_residual: f64,
    /// Relative-residual target the solve was asked for.
    pub tolerance: f64,
    /// Which preconditioner steered the iteration.
    pub preconditioner: ReducedSchurCgPreconditioner,
}

impl ReducedSchurCgReport {
    /// True iff the returned iterate met the requested relative-residual bound.
    /// A `false` here means the iterate is a TRUNCATION, not a solve.
    pub fn converged(&self) -> bool {
        self.relative_residual <= self.tolerance
    }

    /// Merge two certificates into the weaker of the pair, so a bundle of
    /// solves reports its LEAST converged member rather than its best.
    pub fn weaker(self, other: Self) -> Self {
        if other.slack() > self.slack() { other } else { self }
    }

    /// `relative_residual / tolerance`: an exact solve reads 0, and a residual against
    /// a zero tolerance reads unbounded (#2469).
    fn slack(&self) -> f64 {
        if self.relative_residual == 0.0 {
            0.0
        } else {
            self.relative_residual / self.tolerance
        }
    }
}

/// Which preconditioner a reduced-Schur CG solve ran with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReducedSchurCgPreconditioner {
    /// No usable diagonal was available; the iteration ran on the raw operator.
    Identity,
    /// `diag(H_ββ + ridge)` — the shared block's own diagonal, read straight
    /// off the assembled system at zero build cost.
    SharedBlockDiagonal,
}

/// Diagonal preconditioner for the matrix-free reduced-Schur CG.
///
/// `S = (H_ββ + ρ_β I) − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)` is SPD, and its
/// diagonal spans whatever range the shared block's diagonal spans. On the
/// overcomplete SAE border that range is the atom FIRING-COUNT distribution:
/// `H_ββ`'s per-atom diagonal accumulates `Σ_{i ∋ k} φ_i,b²` over the rows on
/// atom `k`'s support, so a dictionary whose atoms fire in 3 rows and 3,000
/// rows carries three orders of magnitude of diagonal spread. Unpreconditioned
/// CG's convergence rate is governed by `√κ(S)`, so that spread alone stalls
/// it — which is exactly the #2576 stagnation (16 probes × 3 groups × the full
/// 4096-iteration cap, tolerance-insensitive because the tolerance was never
/// the binding constraint).
///
/// Against a GENERIC cross-block operator the exact reduced-Schur diagonal needs
/// the point-elimination quotient `Σ_i (H_tβ^(i)e_a)ᵀ(H_tt^(i))⁻¹(H_tβ^(i)e_a)`
/// per column — the `O(n·K)` probe build the Newton-side scalar Jacobi pays,
/// which at the massive-K border costs orders of magnitude more than the solve
/// it would precondition. The SHARED-BLOCK diagonal is already assembled
/// (`hbb_diag` / `penalty_op`), so it is free, and it carries the whole
/// firing-count spread. It is an upper bound on the true diagonal (the
/// eliminated term is PSD), hence strictly positive whenever the assembled
/// diagonal is, and it needs no factorization.
///
/// On the SAE support lane the exact diagonal is affordable (it reads off the
/// staged residency for less than the cost of one matvec), and **it was measured
/// not to help**: 5189 shifted-CG iterations against this shared block's 5138,
/// all tiers agreeing on `log|S|` to 10 significant figures (#2576). The reason
/// is structural, not a tuning accident — `diag(H_ββ)` and the eliminated term
/// are sums over the SAME rows with the SAME `φ²` weights, so the firing-count
/// spread appears in both and cancels, leaving a near-uniform rescaling that CG
/// is invariant to. This shared block is therefore the RIGHT preconditioner, not
/// a cheap stand-in for one.
///
/// This is a preconditioner, not a change of operator: PCG converges to the
/// same `S⁻¹b` as CG, only faster, so every downstream criterion value is
/// unchanged up to the residual tolerance both must meet.
struct ReducedSchurDiagonalPreconditioner {
    inverse_diagonal: Option<Array1<f64>>,
}

/// The shared-block diagonal `diag(H_ββ) + ρ_β` of the reduced Schur, as the
/// preconditioner the SHIFTED rational-surrogate solves take.
///
/// Same diagonal, same justification as
/// [`ReducedSchurDiagonalPreconditioner`] — but the surrogate solves
/// `(S + t_ℓ I)` rather than `S`, and `diag(S + t I) = diag(S) + t`, so one
/// diagonal serves the entire shift ladder with the shift added per solve.
/// This is where the log-determinant lane's iterations actually go: `m` probes
/// times the quadrature's node count, every one an unshifted-to-tiny-shift CG
/// on the same wide-diagonal border (#2576).
///
/// Device seam: this costs no transfers even when the `S·v` apply is running on
/// a GPU. The shifted CG already materializes its residual host-side (the
/// matvec seam hands back an owned `Array1`), so the preconditioner is one
/// elementwise `O(k)` pass over a vector that was already there. It is also
/// reduction-free, hence bit-identical run to run regardless of thread count —
/// the property the criterion's reproducibility contract needs.
pub(crate) fn reduced_schur_shifted_preconditioner(
    sys: &ArrowSchurSystem,
    ridge_beta: f64,
) -> ShiftedDiagonalPreconditioner {
    match ReducedSchurDiagonalPreconditioner::shared_block_diagonal(sys, ridge_beta) {
        Some(diagonal) => ShiftedDiagonalPreconditioner::from_operator_diagonal(&diagonal),
        None => ShiftedDiagonalPreconditioner::identity(),
    }
}

impl ReducedSchurDiagonalPreconditioner {
    /// `diag(H_ββ) + ρ_β`, or `None` when the assembled system carries no
    /// strictly positive finite diagonal to scale by.
    fn shared_block_diagonal(sys: &ArrowSchurSystem, ridge_beta: f64) -> Option<Array1<f64>> {
        if sys.k == 0 {
            return None;
        }
        let mut diag = sys.shared_block_diagonal();
        for value in diag.iter_mut() {
            *value += ridge_beta;
            if !(value.is_finite() && *value > 0.0) {
                return None;
            }
        }
        Some(diag)
    }

    /// A shared block that assembled no diagonal (or a non-positive /
    /// non-finite entry, which the eliminated PSD term can only make worse) has
    /// nothing to scale by: fall back to the identity rather than fabricating a
    /// scale. `S` is still SPD, so plain CG remains correct — just slower,
    /// exactly as before this preconditioner existed.
    fn build(sys: &ArrowSchurSystem, ridge_beta: f64) -> Self {
        Self {
            inverse_diagonal: Self::shared_block_diagonal(sys, ridge_beta)
                .map(|diagonal| diagonal.mapv(|value| 1.0 / value)),
        }
    }

    fn kind(&self) -> ReducedSchurCgPreconditioner {
        match self.inverse_diagonal {
            Some(_) => ReducedSchurCgPreconditioner::SharedBlockDiagonal,
            None => ReducedSchurCgPreconditioner::Identity,
        }
    }

    fn apply(&self, residual: &Array1<f64>) -> Array1<f64> {
        match &self.inverse_diagonal {
            Some(inverse) => residual * inverse,
            None => residual.clone(),
        }
    }
}

/// Preconditioned CG solve `S y = b` on the SPD reduced Schur through the
/// matrix-free [`schur_matvec`] apply (the `t = 0`, unshifted companion to the
/// surrogate's shifted solves), warm-started from `y0`. Yields `y = S⁻¹ b` —
/// the operator every `tr(S⁻¹·M)` gradient / adjoint channel contracts against
/// at massive K — together with the [`ReducedSchurCgReport`] certifying what
/// residual it actually reached.
///
/// `None` on a non-finite breakdown (SPD `S` ⇒ that signals a caller bug or a
/// non-finite operator, both of which must surface rather than be swallowed).
/// Running out of iterations is NOT a breakdown: the iterate is returned with
/// `converged() == false` so the caller decides.
fn reduced_schur_cg_solve<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    b: &Array1<f64>,
    y0: &Array1<f64>,
    cg_rel_tol: f64,
    cg_max_iters: usize,
) -> Option<(Array1<f64>, ReducedSchurCgReport)> {
    // One resident operator reused across every CG apply of this solve — device
    // seam threaded so the inverse-subspace S⁻¹·probe solves ride the resident op.
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    let apply = |v: &Array1<f64>| -> Array1<f64> { op.apply_owned(v) };
    let precond = ReducedSchurDiagonalPreconditioner::build(sys, ridge_beta);
    let quotient = sys.beta_gauge_quotient.as_ref();
    // The preconditioned direction must stay in the quotient complement, or the
    // iteration re-injects the pinned gauge orbit the projected operator has no
    // curvature along. Project inside the preconditioner apply, not after it.
    let precondition = |residual: &Array1<f64>| -> Array1<f64> {
        let z = precond.apply(residual);
        match quotient {
            Some(quotient) => quotient.project_complement(z.view()),
            None => z,
        }
    };
    let b = match quotient {
        Some(quotient) => quotient.project_complement(b.view()),
        None => b.clone(),
    };
    let mut y = match quotient {
        Some(quotient) => quotient.project_complement(y0.view()),
        None => y0.clone(),
    };
    let mut r = &b - &apply(&y);
    // A zero right-hand side has the exact solve `y = 0` whatever the warm start,
    // and past this return `tol = cg_rel_tol * b_norm > 0` (#2469).
    let b_norm = b.dot(&b).sqrt();
    if b_norm == 0.0 {
        return Some((
            Array1::<f64>::zeros(b.len()),
            ReducedSchurCgReport {
                iterations: 0,
                max_iterations: cg_max_iters,
                relative_residual: 0.0,
                tolerance: cg_rel_tol,
                preconditioner: precond.kind(),
            },
        ));
    }
    // One matvec buffer reused across every CG iteration. `apply_owned` builds a
    // fresh `Array1::zeros(k)` per call, which at this scale is an ~11 MB
    // mmap/munmap pair with first-touch faults and a TLB shootdown EVERY
    // iteration -- and this solve runs to its iteration cap on the LAML path.
    // Reuse is the contract `schur_matvec` already documents and enforces: it
    // accumulates, so it clears `out` itself, which also makes the `zeros()`
    // inside `apply_owned` a second redundant zeroing of a buffer about to be
    // discarded.
    let mut ap = Array1::<f64>::zeros(b.len());
    let mut z = precondition(&r);
    let mut p = z.clone();
    let mut rs = r.dot(&z);
    let mut residual_norm_sq = r.dot(&r);
    if !(rs.is_finite() && residual_norm_sq.is_finite()) {
        return None;
    }
    let tol = cg_rel_tol * b_norm;
    let mut iters = 0usize;
    while residual_norm_sq.sqrt() > tol && iters < cg_max_iters {
        op.apply_into(&p, &mut ap);
        let denom = p.dot(&ap);
        if !(denom.is_finite() && denom > 0.0) {
            return None;
        }
        // `rs = rᵀM⁻¹r` is zero only when `r` is, and a zero residual exits
        // through the loop condition above (`tol > 0` always). Reaching here
        // with `rs == 0` therefore means round-off has destroyed the
        // SPD-by-construction preconditioned inner product, and the direction
        // update below would be a division by zero rather than a descent step.
        if rs == 0.0 {
            return None;
        }
        let alpha = rs / denom;
        y.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        residual_norm_sq = r.dot(&r);
        z = precondition(&r);
        let rs_new = r.dot(&z);
        if !(rs_new.is_finite() && residual_norm_sq.is_finite()) {
            return None;
        }
        // In place: `&z + &(&p * c)` allocates two more full-length temporaries
        // per iteration for the same arithmetic.
        p *= rs_new / rs;
        p += &z;
        rs = rs_new;
        iters += 1;
    }
    let report = ReducedSchurCgReport {
        iterations: iters,
        max_iterations: cg_max_iters,
        relative_residual: residual_norm_sq.sqrt() / b_norm,
        tolerance: cg_rel_tol,
        preconditioner: precond.kind(),
    };
    let solved = match quotient {
        Some(quotient) => quotient.project_complement(y.view()),
        None => y,
    };
    Some((solved, report))
}

/// Matrix-free single-rhs reduced-Schur solve `S⁻¹ rhs` (`t = 0`) via CG on
/// `schur_matvec`, warm-started from `warm` (or cold). The base primitive for
/// the selected-inverse gradient channels whose `S⁻¹` argument is NOT the fixed
/// probe family but a per-call probe-derived vector (e.g. `(H⁻¹)_tt`'s
/// `H_βt(H_tt)⁻¹z` term in the ARD latent-block diagonal, and the per-row
/// `(H⁻¹)_tβ` blocks the θ-adjoint / assignment-strength traces contract) — those
/// cannot reuse the `(probes, S⁻¹·probes)` bundle, so they solve `S⁻¹` on demand
/// through this. `None` on a CG breakdown (SPD `S` forbids it, so it signals a
/// non-finite operator or caller bug).
pub fn reduced_schur_inverse_apply<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    rhs: &Array1<f64>,
    warm: Option<&Array1<f64>>,
    cg_rel_tol: f64,
    cg_max_iters: usize,
) -> Option<(Array1<f64>, ReducedSchurCgReport)> {
    let zero = Array1::<f64>::zeros(sys.k);
    let y0 = warm.unwrap_or(&zero);
    reduced_schur_cg_solve(
        sys,
        htt_factors,
        ridge_beta,
        backend,
        resident,
        gpu_matvec,
        rhs,
        y0,
        cg_rel_tol,
        cg_max_iters,
    )
}

fn matrix_free_cache_factor_slab(cache: &ArrowFactorCache) -> &ArrowFactorSlab {
    match &cache.htt_factors_undamped {
        ArrowUndampedFactors::SameAsDamped => &cache.htt_factors,
        ArrowUndampedFactors::Owned(factors) => factors,
    }
}

fn validate_matrix_free_arrow_pair(
    sys: &ArrowSchurSystem,
    cache: &ArrowFactorCache,
    operation: &str,
) -> Result<(), ArrowSchurError> {
    if cache.ridge_t != 0.0 || cache.ridge_beta != 0.0 || !cache.schur_factor_is_undamped {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "{operation} requires an undamped evidence cache; got ridge_t={}, \
                 ridge_beta={}, schur_factor_is_undamped={}",
                cache.ridge_t, cache.ridge_beta, cache.schur_factor_is_undamped
            ),
        });
    }
    if sys.k != cache.k
        || sys.rows.len() != cache.n_rows()
        || sys.row_dims.as_ref() != cache.row_dims.as_ref()
        || sys.row_offsets.as_ref() != cache.row_offsets.as_ref()
    {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "{operation} system/cache layout mismatch: system (rows={}, k={}, offsets={:?}) \
                 vs cache (rows={}, k={}, offsets={:?})",
                sys.rows.len(),
                sys.k,
                sys.row_offsets,
                cache.n_rows(),
                cache.k,
                cache.row_offsets,
            ),
        });
    }
    if sys.row_hessian_fingerprint != cache.row_hessian_fingerprint
        || sys.manifold_mode_fingerprint != cache.manifold_mode_fingerprint
    {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "{operation} refuses a stale matrix-free system/cache pair \
                 (row fingerprint {} vs {}, manifold fingerprint {} vs {})",
                sys.row_hessian_fingerprint,
                cache.row_hessian_fingerprint,
                sys.manifold_mode_fingerprint,
                cache.manifold_mode_fingerprint,
            ),
        });
    }
    if !cache.htbeta_available() && cache.k > 0 {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!("{operation} requires the cached H_tbeta operator"),
        });
    }
    Ok(())
}

fn cholesky_factor_operator_apply(
    factor: ArrayView2<'_, f64>,
    vector: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let n = factor.nrows();
    let mut transposed = Array1::<f64>::zeros(n);
    for col in 0..n {
        let mut value = 0.0_f64;
        for row in col..n {
            value += factor[[row, col]] * vector[row];
        }
        transposed[col] = value;
    }
    let mut out = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut value = 0.0_f64;
        for col in 0..=row {
            value += factor[[row, col]] * transposed[col];
        }
        out[row] = value;
    }
    out
}

/// Apply the undamped full bordered-arrow evidence operator without forming its
/// dense reduced Schur complement.
///
/// The cache supplies the authoritative conditioned row factors and `H_tbeta`
/// operator. The system supplies the matrix-free shared block. Rather than read
/// raw `H_betabeta` directly, this reconstructs it from
/// `S + H_betat A^-1 H_tbeta`, where `S` is applied through the same quotient-
/// aware reduced operator used by the matrix-free log-determinant. Value,
/// selected-inverse traces, and this IFT operator therefore describe one `B`.
pub fn matrix_free_arrow_operator_apply(
    sys: &ArrowSchurSystem,
    cache: &ArrowFactorCache,
    vector_t: ArrayView1<'_, f64>,
    vector_beta: ArrayView1<'_, f64>,
) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
    validate_matrix_free_arrow_pair(sys, cache, "matrix_free_arrow_operator_apply")?;
    if vector_t.len() != cache.delta_t_len() || vector_beta.len() != cache.k {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: format!(
                "matrix_free_arrow_operator_apply vector shapes (t={}, beta={}) != ({}, {})",
                vector_t.len(),
                vector_beta.len(),
                cache.delta_t_len(),
                cache.k,
            ),
        });
    }

    let factors = matrix_free_cache_factor_slab(cache);
    let backend = CpuBatchedBlockSolver;
    let reduced = ReducedSchurOperator::new(sys, factors, 0.0, &backend, None);
    let mut out_beta = reduced.apply(vector_beta);
    let mut out_t = Array1::<f64>::zeros(cache.delta_t_len());
    for row in 0..cache.n_rows() {
        let dim = cache.row_dims[row];
        let start = cache.row_offsets[row];
        let row_vector = vector_t.slice(ndarray::s![start..start + dim]);
        let factor = cache.undamped_factor(row);
        let row_applied = cholesky_factor_operator_apply(factor, row_vector);
        for axis in 0..dim {
            out_t[start + axis] = row_applied[axis];
        }

        if cache.k == 0 {
            continue;
        }
        let mut cross = Array1::<f64>::zeros(dim);
        if !cache.apply_htbeta_row(row, vector_beta, &mut cross) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!("matrix_free_arrow_operator_apply H_tbeta row {row} apply failed"),
            });
        }
        for axis in 0..dim {
            out_t[start + axis] += cross[axis];
        }
        if !cache.apply_htbeta_row_transpose(row, row_vector, &mut out_beta, None) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!("matrix_free_arrow_operator_apply H_betat row {row} apply failed"),
            });
        }

        // `out_beta` already contains `S * vector_beta`; add the eliminated
        // `H_betat A^-1 H_tbeta * vector_beta` term to recover H_betabeta.
        let solved_cross = cholesky_solve_vector(factor, cross.view());
        if !cache.apply_htbeta_row_transpose(row, solved_cross.view(), &mut out_beta, None) {
            return Err(ArrowSchurError::SchurFactorFailed {
                reason: format!(
                    "matrix_free_arrow_operator_apply Schur reconstruction row {row} failed"
                ),
            });
        }
    }
    Ok((out_t, out_beta))
}

/// The `S⁻¹ v_j` bundle for a fixed probe set: solves `S y_j = v_j` (`t = 0`) on
/// the matrix-free reduced Schur for each probe `v_j`, warm-started per-probe
/// from `warm` when supplied (e.g. the surrogate's smallest-shift solves, which
/// already sit close to `S⁻¹ v_j`). Computed ONCE per outer solve and reused
/// across every `tr(S⁻¹·M)` channel, so the whole massive-K ρ-gradient +
/// θ-adjoint rides on one probe family — one functional, desync closed.
///
/// `probes` are the surrogate plan's Rademacher probes (`RationalLogdetPlan::
/// probes`); pass the SAME set the value used so the trace estimates are
/// consistent with it. `None` on any CG breakdown.
///
/// The returned [`ReducedSchurCgReport`] is the bundle's WEAKEST member — a
/// bundle is only as certified as its least-converged solve, and every trace
/// estimated from it averages over all of them.
pub(crate) fn reduced_schur_inverse_probe_solves<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    resident: Option<&SaeResidentReducedSchur>,
    gpu_matvec: Option<&GpuSchurMatvec>,
    probes: &[Array1<f64>],
    warm: Option<&[Array1<f64>]>,
    cg_rel_tol: f64,
    cg_max_iters: usize,
) -> Option<(Vec<Array1<f64>>, ReducedSchurCgReport)> {
    let k = sys.k;
    let zero = Array1::<f64>::zeros(k);
    let mut out = Vec::with_capacity(probes.len());
    let mut weakest: Option<ReducedSchurCgReport> = None;
    for (j, v) in probes.iter().enumerate() {
        let y0 = warm.and_then(|w| w.get(j)).unwrap_or(&zero);
        let (y, report) = reduced_schur_cg_solve(
            sys,
            htt_factors,
            ridge_beta,
            backend,
            resident,
            gpu_matvec,
            v,
            y0,
            cg_rel_tol,
            cg_max_iters,
        )?;
        weakest = Some(match weakest {
            Some(previous) => previous.weaker(report),
            None => report,
        });
        out.push(y);
    }
    weakest.map(|report| (out, report))
}

/// Hutchinson estimate `tr(S⁻¹ M) ≈ (1/m) Σ_j (S⁻¹ v_j)ᵀ (M v_j)` for the reduced
/// Schur `S` and a SYMMETRIC channel operator `M` supplied by its matvec
/// `m_matvec(v) = M·v`. `sinv_probes[j] = S⁻¹ v_j` is the bundle from
/// `reduced_schur_inverse_probe_solves` and `probes` the matching probe set.
///
/// The general umbrella (#2080): every dense-`S⁻¹` consumer in the SAE outer
/// gradient — the per-row selected-inverse deflation corrections
/// (`M = Σ_i G_iᵀ C_i G_i`), the direct β–β contractions (`M = ∂H_ββ` channel),
/// and the θ-adjoint — is ultimately a `tr(S⁻¹·M)` with `M·v` computable
/// row-locally without forming `M`. Estimating them all from the SAME
/// `(probes, S⁻¹ v_j)` pair keeps the value, ρ-gradient, and θ-adjoint one
/// functional. Unbiased for the ±1 Rademacher probes (`E[vᵀ S⁻¹ M v] =
/// tr(S⁻¹ M)`). `None` on a length mismatch or a non-finite accumulation.
pub fn hutchinson_reduced_schur_inverse_trace(
    probes: &[Array1<f64>],
    sinv_probes: &[Array1<f64>],
    m_matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
) -> Option<f64> {
    let m = probes.len();
    if m == 0 || sinv_probes.len() != m {
        return None;
    }
    let mut acc = 0.0_f64;
    for (v, y) in probes.iter().zip(sinv_probes) {
        let mv = m_matvec(v.view());
        acc += y.dot(&mv);
    }
    acc /= m as f64;
    acc.is_finite().then_some(acc)
}

/// Accumulate one row's reduced-Schur point-elimination contribution
/// `H_βt^(i) (H_tt^(i))⁻¹ H_tβ^(i) x` (length `K`) into `acc`.
///
/// `local` is caller-owned `≥ sys.d`-length scratch (reused across rows to keep
/// the hot loop allocation-free); only `..di` is touched. `acc` is **added to**,
/// never cleared, so the caller controls whether contributions sum into a chunk
/// partial (parallel path) or a per-row buffer (sequential path).
#[inline]
pub(crate) fn schur_matvec_row_into<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    x: &Array1<f64>,
    backend: &B,
    i: usize,
    local: &mut Array1<f64>,
    acc: &mut Array1<f64>,
) {
    let row = &sys.rows[i];
    let di = sys.row_dims[i];
    // H_tβ^(i) · x → local[..di], routed through sys.htbeta_matvec
    // when the dense block is absent.
    let mut local_i = local.slice_mut(ndarray::s![..di]).to_owned();
    local_i.fill(0.0);
    sys_htbeta_apply_row(sys, i, row, x.view(), &mut local_i);
    let solved = backend.solve_block_vector(htt_factors.factor(i), local_i.view());
    // H_βt^(i) · solved accumulates into acc (length k).  Routed through
    // sys.htbeta_matvec when needed.
    sys_htbeta_accumulate_transpose(sys, i, row, solved.view(), acc);
}

/// One per-term block factor for the block-Jacobi Schur preconditioner.
///
/// Carries either a dense Cholesky factor (for PD blocks ≤ 256 columns) or
/// the scalar inverses for that block's diagonal as a fallback.
#[derive(Clone)]
pub(crate) enum BlockFactor {
    /// Cholesky L stored column-major via faer. `range` identifies the
    /// columns in the full K-vector this block covers.
    Chol {
        factor: FaerLlt<f64>,
        range: Range<usize>,
    },
    /// Scalar fallback: per-element `1/s_aa` for each column in `range`.
    Scalar {
        inv: Array1<f64>,
        range: Range<usize>,
    },
}

impl std::fmt::Debug for BlockFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockFactor::Chol { range, .. } => {
                write!(f, "BlockFactor::Chol {{ range: {:?} }}", range)
            }
            BlockFactor::Scalar { inv, range } => {
                write!(
                    f,
                    "BlockFactor::Scalar {{ inv.len: {}, range: {:?} }}",
                    inv.len(),
                    range
                )
            }
        }
    }
}

/// Block-Jacobi Schur preconditioner for BA's inexact reduced-system PCG.
///
/// When [`ArrowSchurSystem::block_offsets`] is populated (via
/// [`ArrowSchurSystem::set_block_offsets`]) and the largest block has ≤ 256
/// columns, builds one small dense Schur block per term, factors it with
/// Cholesky (faer LLT), and applies the preconditioner as per-block
/// triangular solves.  Non-PD blocks fall back to scalar diagonal inversion
/// for that block only.  When `block_offsets` is empty or the largest block
/// exceeds 256 columns the preconditioner reduces to pure scalar-diagonal
/// Jacobi (pre-#283 behaviour), so callers that have not called
/// `set_block_offsets` are unaffected.
///
/// The `block_offsets` plumbing is compatible with issue #287 (custom
/// `ParameterBlockSpec` families): those callers supply ranges derived from
/// their own block layout.
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner {
    pub(crate) blocks: Vec<BlockFactor>,
}

/// Maximum block size for which we attempt dense block-Jacobi factorization.
pub(crate) const BLOCK_JACOBI_MAX_BLOCK: usize = 256;

impl JacobiPreconditioner {
    /// Build the block-Jacobi (or scalar fallback) preconditioner from the
    /// Arrow-Schur system without materializing the full dense Schur
    /// complement.
    ///
    /// When `sys.block_offsets` is non-empty and `max(block_size) ≤ 256`,
    /// each block gets a dense `b×b` Schur sub-matrix formed, factored, and
    /// stored.  Otherwise every column gets its own scalar entry.
    pub(crate) fn from_arrow_schur<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        resident: Option<&SaeResidentReducedSchur>,
    ) -> Result<Self, ArrowSchurError> {
        let use_block = !sys.block_offsets.is_empty()
            && sys
                .block_offsets
                .iter()
                .map(|r| r.end.saturating_sub(r.start))
                .max()
                .unwrap_or(0)
                <= BLOCK_JACOBI_MAX_BLOCK;
        if use_block {
            if let Some(res) = resident {
                Self::build_block_jacobi_resident(sys, ridge_beta, res)
            } else {
                Self::build_block_jacobi(sys, htt_factors, ridge_beta, backend)
            }
        } else if let Some(res) = resident {
            // #1017 — SAE residency scalar Jacobi. The generic scalar build
            // probes `H_tβ^(i) e_a` and re-solves `(H_tt^(i))⁻¹` once for EVERY
            // (row, β-column) pair: `O(n·K)` triangular solves and `O(n·K·p)`
            // operator-probe work per Newton step, with `K = K_atoms·p` in the
            // tens of thousands at LLM shapes. The reduced-Schur diagonal is the
            // same quotient the resident `(L_i, Y_i)` factors already carry, so
            // read the diagonal straight off them in one support-sparse pass —
            // no probe, no per-column solve.
            Self::build_scalar_jacobi_resident(sys, ridge_beta, res)
        } else {
            Self::build_scalar_jacobi(sys, htt_factors, ridge_beta, backend)
        }
    }

    /// Build scalar-diagonal Jacobi: one `BlockFactor::Scalar` of length 1
    /// per column.  Matches pre-#283 semantics.
    ///
    /// When `sys.htbeta_matvec` is set and per-row `htbeta` slabs are absent,
    /// each column is probed via the matvec (one call per column per row).
    pub(crate) fn build_scalar_jacobi<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        let k = sys.k;
        // Extract diagonal of H_ββ via penalty_diagonal_add (#296):
        // no Arc-clone; falls back to hbb_diag or hbb[[a,a]] inline.
        let mut diag = Array1::<f64>::zeros(k);
        {
            let diag_slice = diag.as_slice_mut().expect("diag must be contiguous");
            sys.penalty_diagonal_add(diag_slice);
        }
        for a in 0..k {
            diag[a] += ridge_beta;
        }
        // Per-row body: subtract this row's `Σ_a (H_tβ^(i)e_a)ᵀ(H_tt^(i))⁻¹
        // (H_tβ^(i)e_a)` contribution into a caller-provided length-`K` diagonal
        // accumulator (`-=`). For each column `a`, probe the cross-block (or read
        // the dense slab) and compute the scalar point-elimination quotient. The
        // `O(K)` solves per row are the build's whole cost; the row contributions
        // are independent length-`K` vectors, so a worker sums a chunk into a
        // private `diag_part` and the caller folds the partials back in chunk
        // order — bit-identical run-to-run (the #1017 preconditioner gate).
        let row_into = |i: usize, row: &ArrowRowBlock, diag_part: &mut Array1<f64>| {
            let di = sys.row_dims[i];
            // Dense-slab fast path (#1017): when the per-row cross-block is a
            // materialized `di × k` slab (no matrix-free operator), the entire
            // reduced-Schur diagonal contribution for this row is
            // `Σ_c H_tβ[c,a] · ((H_tt)⁻¹ H_tβ)[c,a]`. The generic loop below
            // re-solved `(H_tt)⁻¹` once PER COLUMN — `O(k)` block solves + `O(k)`
            // allocations per row, i.e. `O(n·k)` tiny solves per Newton step
            // (the dominant fixed per-solve cost at the SAE wide-border shape,
            // k in the tens of thousands). Solve all `k` columns in ONE batched
            // block solve instead, then take the column dots. Reassociates the
            // diagonal within the documented #1211 preconditioner margin (same as
            // the resident no-probe path), and the preconditioner only steers the
            // PCG iterate, which still terminates at the PCG tolerance.
            if sys.htbeta_matvec.is_none() && row.htbeta.dim() == (di, k) {
                let solved = backend.solve_block_matrix(htt_factors.factor(i), row.htbeta.view());
                for a in 0..k {
                    let mut acc = 0.0;
                    for c in 0..di {
                        acc += row.htbeta[[c, a]] * solved[[c, a]];
                    }
                    diag_part[a] -= acc;
                }
                return;
            }
            // Matrix-free path: probe column a. `e_a` stays all-zero between
            // columns — set the single active entry and reset it after the probe,
            // so we never pay the `O(k)` `e_a.fill(0.0)` per column (that fill was
            // `O(n·k²)`). `sys_htbeta_apply_row` zeroes `col_i` internally.
            let mut col_i = Array1::<f64>::zeros(di);
            let mut e_a = Array1::<f64>::zeros(k);
            for a in 0..k {
                e_a[a] = 1.0;
                sys_htbeta_apply_row(sys, i, row, e_a.view(), &mut col_i);
                e_a[a] = 0.0;
                let solved = backend.solve_block_vector(htt_factors.factor(i), col_i.view());
                let mut acc = 0.0;
                for c in 0..di {
                    acc += col_i[c] * solved[c];
                }
                diag_part[a] -= acc;
            }
        };
        let n = sys.rows.len();
        let parallel =
            n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            use rayon::prelude::*;
            const CHUNK: usize = 64;
            let partials: Vec<Array1<f64>> = (0..n)
                .into_par_iter()
                .chunks(CHUNK)
                .map(|idxs| {
                    let mut diag_part = Array1::<f64>::zeros(k);
                    for i in idxs {
                        row_into(i, &sys.rows[i], &mut diag_part);
                    }
                    diag_part
                })
                .collect();
            // Deterministic ordered reduction: fold chunk partials left-to-right.
            for part in &partials {
                for a in 0..k {
                    diag[a] += part[a];
                }
            }
        } else {
            for (i, row) in sys.rows.iter().enumerate() {
                row_into(i, row, &mut diag);
            }
        }
        let mut blocks = Vec::with_capacity(k);
        for a in 0..k {
            let v = diag[a];
            if !v.is_finite() || v <= 0.0 {
                return Err(ArrowSchurError::PcgFailed {
                    reason: format!(
                        "invalid Schur Jacobi diagonal at index {a}: {v}; \
                         operator regularization is required"
                    ),
                });
            }
            blocks.push(BlockFactor::Scalar {
                inv: Array1::from_elem(1, 1.0 / v),
                range: a..a + 1,
            });
        }
        Ok(Self { blocks })
    }

    /// Build scalar-diagonal Jacobi from the pre-staged SAE residency factors
    /// `(L_i, Y_i)` (#1017).
    ///
    /// The generic [`Self::build_scalar_jacobi`] forms each reduced-Schur
    /// diagonal entry `S_aa = H_ββ,aa + ρ − Σ_i (H_tβ^(i) e_a)ᵀ(H_tt^(i))⁻¹(H_tβ^(i) e_a)`
    /// by probing the cross-block operator with the unit vector `e_a` and
    /// re-solving `(H_tt^(i))⁻¹` for every `(row, column)` pair — `O(n·K)`
    /// triangular solves per Newton step. For the SAE Kronecker cross-block the
    /// `a`-th column lives on exactly one active support entry: `a = beta_base + j`
    /// for some `(beta_base, φ) ∈ a_phi[i]` and output channel `j ∈ 0..p`, with
    /// `H_tβ^(i) e_a = φ · L_i[:, j]`. The point-elimination quotient is then
    ///
    /// ```text
    /// (H_tβ^(i) e_a)ᵀ (H_tt^(i))⁻¹ (H_tβ^(i) e_a)
    ///     = φ² · L_i[:, j]ᵀ (H_tt^(i))⁻¹ L_i[:, j]
    ///     = φ² · (L_i[:, j] · Y_i[:, j]),          Y_i := (H_tt^(i))⁻¹ L_i.
    /// ```
    ///
    /// so the whole diagonal is accumulated in ONE support-sparse pass over the
    /// resident factors — no probe, no per-column solve, the staged `Y_i` reused
    /// from the matvec residency. The result is the SAME quotient the generic
    /// path computes (up to float reassociation of the row sum), so the PCG
    /// preconditioner is unchanged up to that f64 margin. Since the preconditioner
    /// only steers the iterate (which still terminates at the PCG tolerance), the
    /// criterion ranking is stable except for candidates within that margin,
    /// where the near-tie winner can flip — not an exact no-move guarantee (#1211).
    pub(crate) fn build_scalar_jacobi_resident(
        sys: &ArrowSchurSystem,
        ridge_beta: f64,
        resident: &SaeResidentReducedSchur,
    ) -> Result<Self, ArrowSchurError> {
        let k = sys.k;
        let p = resident.p;
        let n = resident.rows.len();
        // Seed with diag(H_ββ) + ridge — same penalty source the generic path
        // reads, so the only difference is how the point-elimination term is
        // gathered.
        let mut diag = Array1::<f64>::zeros(k);
        {
            let diag_slice = diag.as_slice_mut().expect("diag must be contiguous");
            sys.penalty_diagonal_add(diag_slice);
        }
        for a in 0..k {
            diag[a] += ridge_beta;
        }
        // Per-row point-elimination diagonal: for each active support entry
        // `(beta_base, φ)` and channel `j`, subtract `φ² · L_i[:, j]·Y_i[:, j]`
        // into `diag[beta_base + j]`. `L_i`/`Y_i` are row-major `di × p`, so the
        // `j`-th column dot is `Σ_r L_i[r·p + j]·Y_i[r·p + j]`.
        //
        // The accumulation is into a SHARED `diag` (rows scatter into overlapping
        // `beta_base + j` columns), so — like the generic `build_scalar_jacobi`
        // and the `schur_matvec` row loop (#1017) — parallelism uses worker-private
        // length-`K` partials folded back in chunk order: each chunk is a
        // contiguous ascending row range and rows within it stay ascending, so the
        // chunk-ordered fold reproduces the serial `row = 0..n` subtraction order
        // bit-for-bit run-to-run (the #1017 determinism gate). Run-to-run
        // bit-identity does not extend to bit-identity with the in-place serial
        // accumulation, so the preconditioner — and any criterion ranking it
        // steers — is stable only up to the chunk-reassociation margin; a near-tie
        // winner inside that margin can flip (#1211).
        // This build runs once per inexact-PCG solve = O(inner-Newton-iters)
        // per fit; at the SAE LLM shape (thousands of rows, wide border `k`) the
        // per-row support sweep is the build's whole cost and was on one core.
        // The per-channel column dot `col_dot[j] = Σ_r L_i[r·p+j]·Y_i[r·p+j]`
        // (the diagonal of `G_i = L_iᵀ(H_tt)⁻¹L_i`) depends ONLY on the row `i`,
        // not on the support entry `(beta_base, φ)`. The previous loop recomputed
        // it once per support entry — a row with `m` active atoms paid `m·p`
        // column dots over `di`. Hoist it: compute the `p` column dots once per
        // row into reusable `col_dot` scratch, then each support entry is a pure
        // scatter `diag[beta_base+j] -= φ²·col_dot[j]`. Bit-for-bit identical:
        // each `col_dot[j]` is the same `r`-ascending sum, and `φ²·col_dot[j]`
        // yields identical bits whether `col_dot[j]` was just computed or cached.
        let row_into = |row: usize, diag_part: &mut [f64], col_dot: &mut [f64]| {
            let rf = &resident.rows[row];
            let di = rf.di;
            if di == 0 {
                return;
            }
            let support = &resident.a_phi[row];
            if support.is_empty() {
                return;
            }
            // `L_i` is the shared `local_jac[row]` slab (#1033) — byte-for-byte
            // the former per-row `rf.l` copy.
            let l_i = &resident.local_jac[row];
            for (j, slot) in col_dot.iter_mut().enumerate().take(p) {
                let mut acc = 0.0_f64;
                for r in 0..di {
                    let idx = r * p + j;
                    acc += l_i[idx] * rf.y[idx];
                }
                *slot = acc;
            }
            for &(beta_base, phi) in support {
                if phi == 0.0 {
                    continue;
                }
                let phi2 = phi * phi;
                for j in 0..p {
                    diag_part[beta_base + j] -= phi2 * col_dot[j];
                }
            }
        };
        let parallel =
            n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            use rayon::prelude::*;
            const CHUNK: usize = 64;
            let partials: Vec<Array1<f64>> = (0..n)
                .into_par_iter()
                .chunks(CHUNK)
                .map(|idxs| {
                    let mut diag_part = Array1::<f64>::zeros(k);
                    let mut col_dot = vec![0.0_f64; p];
                    let slice = diag_part
                        .as_slice_mut()
                        .expect("diag_part must be contiguous");
                    for i in idxs {
                        row_into(i, slice, &mut col_dot);
                    }
                    diag_part
                })
                .collect();
            // Deterministic ordered reduction: fold chunk partials left-to-right
            // (each partial already holds the per-row terms subtracted, so add
            // them into `diag` in chunk order to mirror the serial subtraction).
            for part in &partials {
                for a in 0..k {
                    diag[a] += part[a];
                }
            }
        } else {
            let diag_slice = diag.as_slice_mut().expect("diag must be contiguous");
            let mut col_dot = vec![0.0_f64; p];
            for row in 0..n {
                row_into(row, diag_slice, &mut col_dot);
            }
        }
        let mut blocks = Vec::with_capacity(k);
        for a in 0..k {
            let v = diag[a];
            if !v.is_finite() || v <= 0.0 {
                return Err(ArrowSchurError::PcgFailed {
                    reason: format!(
                        "invalid SAE-resident Schur Jacobi diagonal at index {a}: {v}; \
                         operator regularization is required"
                    ),
                });
            }
            blocks.push(BlockFactor::Scalar {
                inv: Array1::from_elem(1, 1.0 / v),
                range: a..a + 1,
            });
        }
        Ok(Self { blocks })
    }

    /// Build block-Jacobi from the pre-staged SAE residency factors `(L_i, Y_i)`.
    ///
    /// This is the block analogue of [`Self::build_scalar_jacobi_resident`].
    /// When SAE block offsets are small enough to select BetaBlockJacobi (for
    /// example per-atom decoder blocks with `basis_size·p <= 256`), the generic
    /// block builder materializes every row's dense `(d_i × K)` `H_tβ` by probing
    /// the matrix-free operator, then re-solves `(H_tt)⁻¹` for each block column.
    /// The resident factors already carry `G_i = L_iᵀ(H_tt)⁻¹L_i`, so each block
    /// is assembled by scattering only the active support pairs inside that block:
    ///
    /// ```text
    /// S_block -= Σ_i Σ_(s,t in block support) φ_s φ_t · G_i[channel_s, channel_t]
    /// ```
    ///
    /// It computes the same block-diagonal restriction as the generic path, but
    /// avoids the full-row `H_tβ` materialization and per-column triangular solves.
    pub(crate) fn build_block_jacobi_resident(
        sys: &ArrowSchurSystem,
        ridge_beta: f64,
        resident: &SaeResidentReducedSchur,
    ) -> Result<Self, ArrowSchurError> {
        let block_offsets = &sys.block_offsets;
        let p = resident.p;
        let mut schur_blocks: Vec<Array2<f64>> = Vec::with_capacity(block_offsets.len());
        for (block_idx, range) in block_offsets.iter().enumerate() {
            let b = range.end - range.start;
            let mut schur_block = Array2::<f64>::zeros((b, b));
            sys.penalty_block_add(
                BetaBlockId(block_idx),
                block_offsets.as_ref(),
                &mut schur_block,
            );
            for bi in 0..b {
                schur_block[[bi, bi]] += ridge_beta;
            }
            schur_blocks.push(schur_block);
        }

        let row_into = |row: usize, blocks: &mut [Array2<f64>]| {
            let rf = &resident.rows[row];
            let di = rf.di;
            if di == 0 {
                return;
            }
            let support = &resident.a_phi[row];
            if support.is_empty() {
                return;
            }
            // `L_i` is the shared `local_jac[row]` slab (#1033) — byte-for-byte
            // the former per-row `rf.l` copy.
            let l_i = &resident.local_jac[row];
            for (block_idx, range) in block_offsets.iter().enumerate() {
                let block = &mut blocks[block_idx];
                for &(base_left, phi_left) in support {
                    if phi_left == 0.0 {
                        continue;
                    }
                    let left_start = base_left.max(range.start);
                    let left_end = (base_left + p).min(range.end);
                    if left_start >= left_end {
                        continue;
                    }
                    for &(base_right, phi_right) in support {
                        if phi_right == 0.0 {
                            continue;
                        }
                        let right_start = base_right.max(range.start);
                        let right_end = (base_right + p).min(range.end);
                        if right_start >= right_end {
                            continue;
                        }
                        let phi = phi_left * phi_right;
                        for gi in left_start..left_end {
                            let li = gi - range.start;
                            let ch_i = gi - base_left;
                            for gj in right_start..right_end {
                                let lj = gj - range.start;
                                let ch_j = gj - base_right;
                                let mut gij = 0.0_f64;
                                for r in 0..di {
                                    gij += l_i[r * p + ch_i] * rf.y[r * p + ch_j];
                                }
                                block[[li, lj]] -= phi * gij;
                            }
                        }
                    }
                }
            }
        };

        let n = resident.rows.len();
        let parallel =
            n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            let n_blocks = block_offsets.len();
            let block_dims: Vec<usize> = block_offsets.iter().map(|r| r.end - r.start).collect();
            let Ok(()) = fold_row_chunk_partials(
                n,
                || {
                    block_dims
                        .iter()
                        .map(|&b| Array2::<f64>::zeros((b, b)))
                        .collect::<Vec<_>>()
                },
                |local| local.iter_mut().for_each(|block| block.fill(0.0)),
                |i, local| {
                    row_into(i, local);
                    Ok::<(), std::convert::Infallible>(())
                },
                |local| {
                    for bidx in 0..n_blocks {
                        schur_blocks[bidx] += &local[bidx];
                    }
                },
            );
        } else {
            for row in 0..n {
                row_into(row, &mut schur_blocks);
            }
        }

        let mut blocks = Vec::with_capacity(block_offsets.len());
        for (block_idx, range) in block_offsets.iter().enumerate() {
            let b = range.end - range.start;
            let schur_block = &schur_blocks[block_idx];
            let factor_opt = {
                use faer::Side;
                let view = FaerArrayView::new(schur_block);
                FaerLlt::new(view.as_ref(), Side::Lower).ok()
            };
            if let Some(llt) = factor_opt {
                blocks.push(BlockFactor::Chol {
                    factor: llt,
                    range: range.clone(),
                });
            } else {
                let mut inv = Array1::<f64>::zeros(b);
                for bi in 0..b {
                    let v = schur_block[[bi, bi]];
                    if !v.is_finite() || v <= 0.0 {
                        return Err(ArrowSchurError::PcgFailed {
                            reason: format!(
                                "SAE-resident block Jacobi scalar fallback: non-PD diagonal at \
                                 global index {}: {v}; regularization required",
                                range.start + bi
                            ),
                        });
                    }
                    inv[bi] = 1.0 / v;
                }
                blocks.push(BlockFactor::Scalar {
                    inv,
                    range: range.clone(),
                });
            }
        }
        Ok(Self { blocks })
    }

    /// Build term-block Jacobi: one dense `b×b` Schur block per term in
    /// `sys.block_offsets`.
    pub(crate) fn build_block_jacobi<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        let block_offsets = &sys.block_offsets;

        // Initialise every b×b Schur sub-block from H_ββ + ridge·I via
        // penalty_block_add (#296): routes to penalty_op or falls back to
        // hbb / hbb_diag inline without Arc-clone per loop iteration. These are
        // the block-diagonal restrictions of the reduced Schur complement; the
        // per-row cross-block contributions are accumulated in the row sweep
        // below.
        let mut schur_blocks: Vec<Array2<f64>> = Vec::with_capacity(block_offsets.len());
        for (block_idx, range) in block_offsets.iter().enumerate() {
            let b = range.end - range.start;
            let mut schur_block = Array2::<f64>::zeros((b, b));
            sys.penalty_block_add(
                BetaBlockId(block_idx),
                block_offsets.as_ref(),
                &mut schur_block,
            );
            for bi in 0..b {
                schur_block[[bi, bi]] += ridge_beta;
            }
            schur_blocks.push(schur_block);
        }

        // Subtract Schur contributions:
        // S_kk -= H_βt_k^(i) (H_tt^(i))^{-1} H_tβ_k^(i)
        //
        // Materialize each row's (d_i × K) cross-block ONCE and scatter its
        // contribution into every block-diagonal sub-block — mirroring the
        // row-outer structure of `build_dense_schur_direct`. The previous
        // block-outer form re-materialized every row for each β-block
        // (O(n_blocks · n · K) probes); for the matrix-free softmax cross-block
        // each materialize is itself O(K²), so that nesting made the
        // preconditioner build quadratically more expensive than the direct
        // dense Schur it preconditions. sys_htbeta_materialize_row handles the
        // Kronecker / htbeta_matvec path transparently.
        // Per-row body: materialize the row's `(d_i × K)` cross-block once and
        // subtract its `H_βt_k^(i)(H_tt^(i))⁻¹H_tβ_k^(i)` contribution into EACH
        // block-diagonal sub-block. Writes INTO a caller-provided `blocks`
        // accumulator (`-=`) so a rayon worker can subtract a chunk's rows into
        // a worker-private zero-seeded `Vec<Array2>` and the caller folds the
        // chunk partials back in chunk order — bit-identical run-to-run
        // regardless of thread scheduling (the #1017 verification gate). This
        // is deterministic and within the chunk-reassociation margin of serial,
        // so the preconditioner, hence the criterion ranking, is stable except
        // for near-tie candidates inside that f64 margin — not an exact no-move
        // guarantee (#1211).
        let row_into = |i: usize,
                        row: &ArrowRowBlock,
                        blocks: &mut [Array2<f64>]|
         -> Result<(), ArrowSchurError> {
            let di = sys.row_dims[i];
            let htbeta_full = sys_htbeta_materialize_row(sys, i, row)?;
            for (block_idx, range) in block_offsets.iter().enumerate() {
                let b = range.end - range.start;
                let mut solved_cols = Array2::<f64>::zeros((di, b));
                for bj in 0..b {
                    let gj = range.start + bj;
                    let rhs = htbeta_full.column(gj).to_owned();
                    let solved = backend.solve_block_vector(htt_factors.factor(i), rhs.view());
                    for c in 0..di {
                        solved_cols[[c, bj]] = solved[c];
                    }
                }
                let schur_block = &mut blocks[block_idx];
                for bi in 0..b {
                    let gi = range.start + bi;
                    for bj in 0..b {
                        let mut acc = 0.0;
                        for c in 0..di {
                            acc += htbeta_full[[c, gi]] * solved_cols[[c, bj]];
                        }
                        schur_block[[bi, bj]] -= acc;
                    }
                }
            }
            Ok(())
        };
        // Each row materializes an `O(K²)` cross-block (Kronecker) plus `Σ_k b_k`
        // triangular solves — the preconditioner build's whole per-row cost at
        // the SAE LLM shape (#1017), and the rows are independent. Fan over fixed
        // row chunks above the threshold, staying serial for the handful-of-rows
        // non-SAE callers and inside a rayon worker (topology-race nesting guard)
        // — the same gate `schur_matvec` uses.
        let n = sys.rows.len();
        let parallel =
            n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
        if parallel {
            let n_blocks = block_offsets.len();
            let block_dims: Vec<usize> = block_offsets.iter().map(|r| r.end - r.start).collect();
            // Deterministic ordered reduction: fold chunk partials left-to-right.
            fold_row_chunk_partials(
                n,
                || {
                    block_dims
                        .iter()
                        .map(|&b| Array2::<f64>::zeros((b, b)))
                        .collect::<Vec<_>>()
                },
                |local| local.iter_mut().for_each(|block| block.fill(0.0)),
                |i, local| row_into(i, &sys.rows[i], local),
                |local| {
                    for bidx in 0..n_blocks {
                        schur_blocks[bidx] += &local[bidx];
                    }
                },
            )?;
        } else {
            for (i, row) in sys.rows.iter().enumerate() {
                row_into(i, row, &mut schur_blocks)?;
            }
        }

        // Factor each accumulated block: LLT, with scalar-diagonal fallback for
        // a block that comes out non-PD at this ridge.
        let mut blocks = Vec::with_capacity(block_offsets.len());
        for (block_idx, range) in block_offsets.iter().enumerate() {
            let b = range.end - range.start;
            let schur_block = &schur_blocks[block_idx];
            let factor_opt = {
                use faer::Side;
                let view = FaerArrayView::new(schur_block);
                FaerLlt::new(view.as_ref(), Side::Lower).ok()
            };
            if let Some(llt) = factor_opt {
                blocks.push(BlockFactor::Chol {
                    factor: llt,
                    range: range.clone(),
                });
            } else {
                // Non-PD block: fall back to scalar diagonal for this block.
                let mut inv = Array1::<f64>::zeros(b);
                for bi in 0..b {
                    let v = schur_block[[bi, bi]];
                    if !v.is_finite() || v <= 0.0 {
                        return Err(ArrowSchurError::PcgFailed {
                            reason: format!(
                                "block Jacobi scalar fallback: non-PD diagonal at \
                                 global index {}: {v}; regularization required",
                                range.start + bi
                            ),
                        });
                    }
                    inv[bi] = 1.0 / v;
                }
                blocks.push(BlockFactor::Scalar {
                    inv,
                    range: range.clone(),
                });
            }
        }
        Ok(Self { blocks })
    }

    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for block in &self.blocks {
            match block {
                BlockFactor::Scalar { inv, range } => {
                    for (local, gi) in range.clone().enumerate() {
                        out[gi] = inv[local] * r[gi];
                    }
                }
                BlockFactor::Chol { factor, range } => {
                    let b = range.end - range.start;
                    let mut rhs = Array1::<f64>::zeros(b);
                    for (local, gi) in range.clone().enumerate() {
                        rhs[local] = r[gi];
                    }
                    let stride = rhs.strides()[0];
                    let len = rhs.len();
                    // SAFETY: rhs is a uniquely-borrowed contiguous Array1
                    // with positive stride (standard layout).
                    let rhs_mat =
                        unsafe { faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0) };
                    let solved = factor.solve(rhs_mat);
                    for (local, gi) in range.clone().enumerate() {
                        out[gi] = solved[(local, 0)];
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Preconditioner ladder: ClusterJacobi, AdditiveSchwarz  (issue #299)
// ---------------------------------------------------------------------------

/// Escalate beyond BetaBlockJacobi only when K exceeds this value and PCG
/// exhausted `max_iterations`.
pub(crate) const PRECOND_ESCALATE_K_THRESHOLD: usize = 100;

/// #1026 matrix-free Schur curvature-floor (the unbounded-PCG analogue of the
/// dense `spectral_pd_floored_schur`). On `pᵀSp ≤ 0` in the unbounded SAE inner
/// PCG, the operator ridge is lifted by the minimal amount that restores
/// positive curvature along the offending direction, plus this fractional
/// margin (so the next CG iterate sits strictly inside the positive cone, not on
/// the `0` knife-edge).
pub(crate) const SCHUR_CURVATURE_FLOOR_MARGIN: f64 = 1.0e-2;
/// Lower bound on the curvature-floor ridge bump, relative to the rhs scale, so
/// a `pᵀSp` that rounds to exactly `0` still gets a strictly positive bump.
pub(crate) const SCHUR_CURVATURE_FLOOR_REL_FLOOR: f64 = 1.0e-12;
/// Ceiling on the accumulated curvature-floor ridge, relative to the rhs scale.
/// Beyond this the operator is treated as un-conditionable by a minimal floor
/// and the recoverable failure is handed to the outer LM loop (which re-forms
/// the whole system at a heavier ridge). Generous so that a large collapsed
/// over-subtraction `(H_tβ)²/H_tt` is still reachable.
pub(crate) const SCHUR_CURVATURE_FLOOR_REL_CEILING: f64 = 1.0e12;
/// Multiplicative growth for the DIAGONAL-refusal ridge escalation (no
/// `(curvature, ‖p‖²)` deficit is available there), matching the per-row
/// `factor_one_row_result` `RIDGE_GROWTH_FACTOR`.
pub(crate) const SCHUR_CURVATURE_FLOOR_DIAG_GROWTH: f64 = 10.0;
/// Max curvature-floor ridge-lift attempts before deferring to the outer LM
/// loop. The diagonal-refusal path grows ×10 per attempt, so this bounds the
/// reachable ridge at `rhs_scale · 10^(attempts)` — ample for any realistic
/// over-subtraction while still bounded.
pub(crate) const SCHUR_CURVATURE_FLOOR_MAX_ATTEMPTS: usize = 24;

/// Cholesky or scalar factor for one cluster of the beta-coefficient graph.
#[derive(Clone)]
pub(crate) enum ClusterFactor {
    Chol {
        cols: Vec<usize>,
        factor: FaerLlt<f64>,
    },
    Scalar {
        cols: Vec<usize>,
        inv: Vec<f64>,
    },
}

impl std::fmt::Debug for ClusterFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClusterFactor::Chol { cols, .. } => {
                write!(f, "ClusterFactor::Chol {{ cols.len: {} }}", cols.len())
            }
            ClusterFactor::Scalar { cols, inv } => write!(
                f,
                "ClusterFactor::Scalar {{ cols.len: {}, inv.len: {} }}",
                cols.len(),
                inv.len()
            ),
        }
    }
}

/// Maximum columns per cluster before scalar fallback.
pub(crate) const CLUSTER_JACOBI_MAX_CLUSTER: usize = 512;

/// Host-memory budget for ONE cluster's dense reduced-Schur Cholesky factor
/// (the `b×b` f64 `L` the cluster-Jacobi preconditioner stores and applies).
///
/// The co-visibility cluster partition caps a cluster's total column count `b`
/// at the largest value whose factor fits this budget, `b_max = ⌊√(budget/8)⌋`
/// (`8b²` bytes for an `f64` `b×b` factor). This DERIVES the cluster-size cap
/// from the factor's memory footprint rather than asserting a bare number:
/// beyond `b_max` the dense factor's `O(b²)` apply also throttles the CG
/// iteration budget, so the cap is the point past which a single dense block
/// stops being the right preconditioner and the partition must split instead.
/// 2 MiB ⇒ `b_max = 512`, pinned equal to [`CLUSTER_JACOBI_MAX_CLUSTER`] by
/// [`tests::covisibility_cap_is_derived_from_factor_budget`] so the co-visibility
/// partition and the legacy scalar-fallback ceiling agree by construction.
pub(crate) const CLUSTER_SCHUR_FACTOR_BYTES_BUDGET: u128 = 2 * 1024 * 1024;

/// Derived co-visibility cluster-size cap (columns): the largest `b` whose dense
/// `b×b` f64 Cholesky factor fits [`CLUSTER_SCHUR_FACTOR_BYTES_BUDGET`]. See that
/// constant for the memory justification. Never below 1.
pub(crate) fn covisibility_cluster_max_cols() -> usize {
    let b = ((CLUSTER_SCHUR_FACTOR_BYTES_BUDGET / 8) as f64)
        .sqrt()
        .floor() as usize;
    b.max(1)
}

/// Maximum columns in a single connected component for which the IC(0)
/// preconditioner assembles the dense `S[C,C]` to derive its sparsity pattern.
/// IC(0) is cheap to APPLY at any size, but the pattern is read from the dense
/// assembly, which is `O(b²)` memory; beyond this the component falls back to
/// the scalar reciprocal diagonal (the same ceiling concern as
/// `CLUSTER_JACOBI_MAX_CLUSTER`, lifted because the IC(0) FACTOR is sparse).
pub(crate) const IC0_MAX_COMPONENT: usize = 4096;

/// Relative threshold below which an assembled `S[i,j]` is treated as a
/// structural zero when deriving the IC(0) level-0 pattern. Scaled by
/// `sqrt(|S_ii|·|S_jj|)` so it is invariant to column scaling; this prunes
/// entries that are pure FMA round-off (a genuinely decoupled `(i,j)` pair
/// assembles to ~0) so they do not enter the kept fill pattern.
pub(crate) const IC0_PATTERN_REL_DROP: f64 = 1.0e-13;

/// Assemble the dense `b×b` reduced-Schur block for the column set `cols`:
/// `S[cols, cols] = H_ββ[cols, cols] + ridge·I − Σ_i H_tβ[cols]ᵀ (H_tt^i)⁻¹ H_tβ[cols]`.
///
/// Shared by `ClusterJacobiPreconditioner::build_from_column_groups` (which
/// Cholesky-factors the returned block) and `DiagAssembledSchwarzPreconditioner`
/// (which inverts each subdomain block and keeps only its diagonal). The result
/// is the LOWER triangle filled by the row reduction; callers that need the full
/// symmetric block must `symmetrize_upper_from_lower`.
///
/// The per-row Schur contribution is fanned over fixed [`SCHUR_FOLD_ROW_CHUNK`]-row
/// chunks above `SCHUR_MATVEC_PARALLEL_ROW_MIN` and folded in chunk order through
/// [`fold_row_chunk_partials`], exactly as in `build_block_jacobi`: bit-identical
/// run-to-run, and equal to the serial path up to the chunk-boundary
/// reassociation (#1017, #1211).
pub(crate) fn assemble_local_schur_block<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    cols: &[usize],
) -> Array2<f64> {
    let b = cols.len();
    let mut s_block = Array2::<f64>::zeros((b, b));
    // Initialise from H_ββ via penalty_subblock_add (#296): routes through
    // penalty_op or falls back to hbb / hbb_diag inline.
    sys.penalty_subblock_add(cols, &mut s_block);
    for bi in 0..b {
        s_block[[bi, bi]] += ridge_beta;
    }
    let cluster_row_into = |row_idx: usize, row: &ArrowRowBlock, acc: &mut Array2<f64>| {
        // Materialize the b needed cross-block columns through the ROUTED
        // `H_tβ` convention (`sys_htbeta_apply_row`: matrix-free operator plus
        // any dense supplement) at the row's OWN width `di` — never a raw
        // `row.htbeta` read at the global `sys.d`: matvec-backed rows carry
        // absent/zero-sized slabs by contract (a raw read is wrong or panics),
        // and per-row widths vary.
        let di = sys.row_dims[row_idx];
        let mut e_g = Array1::<f64>::zeros(sys.k);
        let mut col_i = Array1::<f64>::zeros(di);
        let mut cols_mat = Array2::<f64>::zeros((di, b));
        let mut solved_cols = Array2::<f64>::zeros((di, b));
        for bj in 0..b {
            let gj = cols[bj];
            e_g[gj] = 1.0;
            sys_htbeta_apply_row(sys, row_idx, row, e_g.view(), &mut col_i);
            e_g[gj] = 0.0;
            let solved = backend.solve_block_vector(htt_factors.factor(row_idx), col_i.view());
            for c in 0..di {
                cols_mat[[c, bj]] = col_i[c];
                solved_cols[[c, bj]] = solved[c];
            }
        }
        for bi in 0..b {
            for bj in 0..b {
                let mut dot = 0.0;
                for c in 0..di {
                    dot += cols_mat[[c, bi]] * solved_cols[[c, bj]];
                }
                acc[[bi, bj]] -= dot;
            }
        }
    };
    let n = sys.rows.len();
    let parallel = n >= SCHUR_MATVEC_PARALLEL_ROW_MIN && rayon::current_thread_index().is_none();
    if parallel {
        let Ok(()) = fold_row_chunk_partials(
            n,
            || Array2::<f64>::zeros((b, b)),
            |local| local.fill(0.0),
            |i, local| {
                cluster_row_into(i, &sys.rows[i], local);
                Ok::<(), std::convert::Infallible>(())
            },
            |local| s_block += local,
        );
    } else {
        for (row_idx, row) in sys.rows.iter().enumerate() {
            cluster_row_into(row_idx, row, &mut s_block);
        }
    }
    s_block
}

/// Column groups for the bounded co-visibility cluster preconditioner.
///
/// Builds the weighted co-firing graph over `sys.block_offsets` and returns the
/// column sets of its bounded co-visibility partition
/// (`BetaCouplingGraph::covisibility_cluster_partition`), each capped at
/// [`covisibility_cluster_max_cols`] columns. With no registered block offsets
/// there is no block structure to cluster, so the whole `0..k` border is one
/// group (identical to the component-partition builders' `block_offsets`-empty
/// case). Each group's columns are sorted ascending.
pub(crate) fn covisibility_column_groups(sys: &ArrowSchurSystem) -> Vec<Vec<usize>> {
    if sys.block_offsets.is_empty() {
        return vec![(0..sys.k).collect()];
    }
    let graph = BetaCouplingGraph::build_from_system(sys);
    graph
        .covisibility_cluster_partition(&sys.block_offsets, covisibility_cluster_max_cols())
        .iter()
        .map(|blocks| {
            let mut cols: Vec<usize> = blocks
                .iter()
                .flat_map(|&b| sys.block_offsets[b].clone())
                .collect();
            cols.sort_unstable();
            cols
        })
        .collect()
}

/// Dense Schur block per connected component of the beta-coupling graph.
///
/// Nodes = beta blocks (`block_offsets`); edges = rows where two blocks
/// co-occur with nonzero `H_t_beta` entries. One Cholesky factor per
/// connected component; applied as a triangular solve.
#[derive(Debug, Clone)]
pub struct ClusterJacobiPreconditioner {
    pub(crate) clusters: Vec<ClusterFactor>,
}

impl ClusterJacobiPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        if sys.block_offsets.is_empty() {
            let cols: Vec<usize> = (0..sys.k).collect();
            return Self::build_from_column_groups(sys, htt_factors, ridge_beta, backend, &[cols]);
        }
        let graph = BetaCouplingGraph::build_from_system(sys);
        let col_groups: Vec<Vec<usize>> = graph
            .component_partition()
            .iter()
            .map(|comp_blocks| {
                let mut cols: Vec<usize> = comp_blocks
                    .iter()
                    .flat_map(|&b| sys.block_offsets[b].clone())
                    .collect();
                cols.sort_unstable();
                cols
            })
            .collect();
        Self::build_from_column_groups(sys, htt_factors, ridge_beta, backend, &col_groups)
    }

    /// Cluster-Jacobi from the bounded CO-VISIBILITY partition (Kushal & Agarwal,
    /// CVPR 2012) — the default above the size cap.
    ///
    /// [`Self::from_arrow_schur`] groups β-blocks by CONNECTED COMPONENT of the
    /// co-firing graph. At real over-complete SAE widths that graph is a single
    /// giant component (transitive co-firing), so the lone component's column
    /// count exceeds [`CLUSTER_JACOBI_MAX_CLUSTER`] and
    /// [`Self::build_from_column_groups`] degrades the whole tier to the scalar
    /// reciprocal diagonal — the scaling ceiling (cross-atom coupling through
    /// co-activating atoms with overlapping ambient subspaces is dropped, and PCG
    /// iteration counts blow up). This builder instead partitions the co-firing
    /// graph into clusters bounded by [`covisibility_cluster_max_cols`], keeping
    /// the strongest co-firing edges inside a cluster, so each cluster's dense
    /// Cholesky conditions the strong cross-atom coupling the scalar diagonal
    /// misses while staying inside the per-factor memory budget.
    ///
    /// With no registered `block_offsets` (or a graph that fits the cap in one
    /// piece) the partition is a single group and this coincides with
    /// [`Self::from_arrow_schur`]. Because the preconditioner only steers the CG
    /// iterate over the SAME reduced operator, the solve converges to the SAME
    /// reduced-system solution regardless of the partition — REML-neutral.
    pub(crate) fn from_arrow_schur_covisibility<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        let col_groups = covisibility_column_groups(sys);
        Self::build_from_column_groups(sys, htt_factors, ridge_beta, backend, &col_groups)
    }

    pub(crate) fn build_from_column_groups<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        col_groups: &[Vec<usize>],
    ) -> Result<Self, ArrowSchurError> {
        let mut clusters = Vec::with_capacity(col_groups.len());
        for cols in col_groups {
            let b = cols.len();
            if b == 0 {
                continue;
            }
            if b > CLUSTER_JACOBI_MAX_CLUSTER {
                let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                clusters.push(ClusterFactor::Scalar {
                    cols: cols.clone(),
                    inv,
                });
                continue;
            }
            let mut s_block =
                assemble_local_schur_block(sys, htt_factors, ridge_beta, backend, cols);
            symmetrize_upper_from_lower(&mut s_block);
            let factor_opt = {
                use faer::Side;
                let view = FaerArrayView::new(&s_block);
                FaerLlt::new(view.as_ref(), Side::Lower).ok()
            };
            if let Some(llt) = factor_opt {
                clusters.push(ClusterFactor::Chol {
                    cols: cols.clone(),
                    factor: llt,
                });
            } else {
                let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                clusters.push(ClusterFactor::Scalar {
                    cols: cols.clone(),
                    inv,
                });
            }
        }
        Ok(Self { clusters })
    }

    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for cluster in &self.clusters {
            apply_cluster(cluster, r, &mut out, &ClusterApplyMode::Overwrite);
        }
        out
    }
}

/// Additive Schwarz: base components expanded by `overlap` graph-hops;
/// overlapping columns averaged by partition-of-unity weights.
#[derive(Debug, Clone)]
pub struct AdditiveSchwarzPreconditioner {
    pub(crate) clusters: Vec<ClusterFactor>,
    pub(crate) weights: Vec<f64>,
}

impl AdditiveSchwarzPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        overlap: usize,
    ) -> Result<Self, ArrowSchurError> {
        if sys.block_offsets.is_empty() {
            let cols: Vec<usize> = (0..sys.k).collect();
            let inner = ClusterJacobiPreconditioner::build_from_column_groups(
                sys,
                htt_factors,
                ridge_beta,
                backend,
                &[cols],
            )?;
            return Ok(Self {
                clusters: inner.clusters,
                weights: vec![1.0f64; sys.k],
            });
        }
        let graph = BetaCouplingGraph::build_from_system(sys);
        let col_groups: Vec<Vec<usize>> = graph
            .component_partition()
            .iter()
            .map(|seed| {
                let mut current = seed.clone();
                for _ in 0..overlap {
                    current = graph.expand_one_hop(&current);
                }
                let mut cols: Vec<usize> = current
                    .iter()
                    .flat_map(|&b| sys.block_offsets[b].clone())
                    .collect();
                cols.sort_unstable();
                cols.dedup();
                cols
            })
            .collect();
        let mut counts = vec![0u32; sys.k];
        for cols in &col_groups {
            for &gi in cols {
                counts[gi] += 1;
            }
        }
        let weights: Vec<f64> = counts
            .iter()
            .map(|&c| if c == 0 { 1.0 } else { 1.0 / c as f64 })
            .collect();
        let inner = ClusterJacobiPreconditioner::build_from_column_groups(
            sys,
            htt_factors,
            ridge_beta,
            backend,
            &col_groups,
        )?;
        Ok(Self {
            clusters: inner.clusters,
            weights,
        })
    }

    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for cluster in &self.clusters {
            apply_cluster(
                cluster,
                r,
                &mut out,
                &ClusterApplyMode::Accumulate {
                    weights: &self.weights,
                },
            );
        }
        out
    }
}

/// Diagonal-assembled additive Schwarz (#299).
///
/// The cheap Schwarz variant the domain-decomposition literature recommends as
/// the default for sparse-coupling β-graphs: instead of storing and applying a
/// dense Cholesky factor per overlapping subdomain (as
/// [`AdditiveSchwarzPreconditioner`] does), it inverts each overlapping
/// subdomain Schur block ONCE at build time and keeps only the **diagonal of the
/// local inverse** `(A_k⁻¹)_ii`. Those per-subdomain diagonal contributions are
/// then assembled additively across overlapping subdomains with partition-of-
/// unity weights into a single global diagonal `m`, applied as `out[i] = m[i]·r[i]`.
///
/// This is strictly richer than scalar Jacobi (`1/S_ii`): the local inverse
/// diagonal `(A_k⁻¹)_ii` folds in the off-diagonal coupling WITHIN the subdomain,
/// so a strongly-coupled column gets a smaller (better-damped) effective scale
/// than its bare reciprocal diagonal would give — while the apply stays `O(K)`
/// (one multiply per column), unlike the `O(Σ b_k²)` triangular solves of dense
/// Schwarz. For `overlap = 0` and one column per subdomain it reduces exactly to
/// scalar Jacobi.
#[derive(Debug, Clone)]
pub(crate) struct DiagAssembledSchwarzPreconditioner {
    /// Global per-column multiplier `m[i]`; `out[i] = m[i] · r[i]`.
    pub(crate) inv_diag: Vec<f64>,
}

impl DiagAssembledSchwarzPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        overlap: usize,
    ) -> Result<Self, ArrowSchurError> {
        // Build the overlapping subdomain column groups exactly like
        // AdditiveSchwarz (component partition + `overlap` graph-hop expansion),
        // so the two Schwarz variants decompose the β space identically and
        // differ only in how each subdomain's local inverse is applied.
        let col_groups: Vec<Vec<usize>> = if sys.block_offsets.is_empty() {
            vec![(0..sys.k).collect()]
        } else {
            let graph = BetaCouplingGraph::build_from_system(sys);
            graph
                .component_partition()
                .iter()
                .map(|seed| {
                    let mut current = seed.clone();
                    for _ in 0..overlap {
                        current = graph.expand_one_hop(&current);
                    }
                    let mut cols: Vec<usize> = current
                        .iter()
                        .flat_map(|&b| sys.block_offsets[b].clone())
                        .collect();
                    cols.sort_unstable();
                    cols.dedup();
                    cols
                })
                .collect()
        };
        Self::build_from_column_groups(sys, htt_factors, ridge_beta, backend, &col_groups)
    }

    pub(crate) fn build_from_column_groups<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
        col_groups: &[Vec<usize>],
    ) -> Result<Self, ArrowSchurError> {
        // Partition-of-unity weights: a column shared by `c` subdomains gets each
        // of its `c` diagonal contributions scaled by `1/c`, so the assembled
        // diagonal is a convex combination (and reduces to a single contribution
        // for non-overlapping columns).
        let mut counts = vec![0u32; sys.k];
        for cols in col_groups {
            for &gi in cols {
                counts[gi] += 1;
            }
        }
        let mut accum = vec![0.0f64; sys.k];
        for cols in col_groups {
            let b = cols.len();
            if b == 0 {
                continue;
            }
            // For large subdomains, the dense inverse is too costly; fall back to
            // the global scalar Schur diagonal inverse `1/S_ii` for those columns
            // (the diag-assembled variant then coincides with scalar Jacobi over
            // that subdomain, which is exactly the intended cheap degradation).
            if b > CLUSTER_JACOBI_MAX_CLUSTER {
                let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                for (local, &gi) in cols.iter().enumerate() {
                    let w = if counts[gi] == 0 {
                        1.0
                    } else {
                        1.0 / counts[gi] as f64
                    };
                    accum[gi] += w * inv[local];
                }
                continue;
            }
            let mut s_block =
                assemble_local_schur_block(sys, htt_factors, ridge_beta, backend, cols);
            symmetrize_upper_from_lower(&mut s_block);
            // Diagonal of the local inverse `(A_k⁻¹)_ii`, obtained by solving
            // `A_k X = I` through the same faer Cholesky used elsewhere; on a
            // non-PD local block, degrade to the scalar reciprocal diagonal.
            let local_inv_diag = match local_inverse_diagonal(&s_block) {
                Some(diag) => diag,
                None => {
                    let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                    inv
                }
            };
            for (local, &gi) in cols.iter().enumerate() {
                let w = if counts[gi] == 0 {
                    1.0
                } else {
                    1.0 / counts[gi] as f64
                };
                accum[gi] += w * local_inv_diag[local];
            }
        }
        // A column never covered by any subdomain (only possible for `k` columns
        // with no block_offsets coverage) keeps a neutral unit scale.
        for (gi, &c) in counts.iter().enumerate() {
            if c == 0 {
                accum[gi] = 1.0;
            }
        }
        for (gi, m) in accum.iter().enumerate() {
            if !m.is_finite() || *m <= 0.0 {
                return Err(ArrowSchurError::PcgFailed {
                    reason: format!(
                        "diag-assembled Schwarz: non-positive assembled diagonal at index {gi}: {m}"
                    ),
                });
            }
        }
        Ok(Self { inv_diag: accum })
    }

    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for (gi, &m) in self.inv_diag.iter().enumerate() {
            out[gi] = m * r[gi];
        }
        out
    }
}

/// Diagonal of `A⁻¹` for a small dense SPD block `A`, via the same faer
/// Cholesky used by the cluster/Schwarz factors. Returns `None` if `A` is not
/// positive-definite (caller degrades to the scalar reciprocal diagonal).
pub(crate) fn local_inverse_diagonal(a: &Array2<f64>) -> Option<Vec<f64>> {
    let b = a.nrows();
    let llt = {
        use faer::Side;
        let view = FaerArrayView::new(a);
        FaerLlt::new(view.as_ref(), Side::Lower).ok()?
    };
    let mut diag = Vec::with_capacity(b);
    for col in 0..b {
        // Solve `A x = e_col`; the `col`-th entry of `x` is `(A⁻¹)_{col,col}`.
        let mut rhs = Array1::<f64>::zeros(b);
        rhs[col] = 1.0;
        let stride = rhs.strides()[0];
        let len = rhs.len();
        // SAFETY: `rhs` is a uniquely-borrowed contiguous `Array1<f64>` of `len`
        // elements with positive row stride; a single column never dereferences
        // the column stride, so `0` is sound.
        let rhs_mat = unsafe { faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0) };
        let solved = llt.solve(rhs_mat);
        diag.push(solved[(col, 0)]);
    }
    Some(diag)
}

/// How a cluster factor's contribution is written into the output vector.
///
/// `Overwrite` assigns `out[gi] = value` (non-overlapping clusters, each global
/// column touched by exactly one cluster). `Accumulate` adds the partition-of-unity
/// weighted contribution `out[gi] += weights[gi] * value` (overlapping Schwarz
/// clusters, where a column may belong to several clusters).
pub(crate) enum ClusterApplyMode<'w> {
    Overwrite,
    Accumulate { weights: &'w [f64] },
}

impl ClusterApplyMode<'_> {
    #[inline]
    pub(crate) fn write(&self, out: &mut Array1<f64>, gi: usize, value: f64) {
        match self {
            ClusterApplyMode::Overwrite => out[gi] = value,
            ClusterApplyMode::Accumulate { weights } => out[gi] += weights[gi] * value,
        }
    }
}

/// Apply a single cluster factor to the residual `r`, writing into `out`
/// according to `mode` (overwrite for non-overlapping clusters, weighted
/// accumulate for overlapping Schwarz clusters).
pub(crate) fn apply_cluster(
    cluster: &ClusterFactor,
    r: &Array1<f64>,
    out: &mut Array1<f64>,
    mode: &ClusterApplyMode<'_>,
) {
    match cluster {
        ClusterFactor::Scalar { cols, inv } => {
            for (local, &gi) in cols.iter().enumerate() {
                mode.write(out, gi, inv[local] * r[gi]);
            }
        }
        ClusterFactor::Chol { cols, factor } => {
            let b = cols.len();
            let mut rhs = Array1::<f64>::zeros(b);
            for (local, &gi) in cols.iter().enumerate() {
                rhs[local] = r[gi];
            }
            let stride = rhs.strides()[0];
            let len = rhs.len();
            // SAFETY: rhs is uniquely-borrowed contiguous Array1 with positive stride.
            let rhs_mat = unsafe { faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0) };
            let solved = factor.solve(rhs_mat);
            for (local, &gi) in cols.iter().enumerate() {
                mode.write(out, gi, solved[(local, 0)]);
            }
        }
    }
}

/// One connected-component factor of the block IC(0) preconditioner.
///
/// `IncompleteChol` holds a sparse lower-triangular `L̃` in column-compressed
/// form over the component's local indices: `col_ptr[j]..col_ptr[j+1]` indexes
/// into `(row_idx, val)` for column `j` (rows `>= j`, diagonal first). `cols`
/// maps a local index back to its global β column. `Scalar` is the non-PD /
/// oversized degradation, identical in meaning to [`ClusterFactor::Scalar`].
#[derive(Clone)]
pub(crate) enum Ic0Factor {
    IncompleteChol {
        cols: Vec<usize>,
        col_ptr: Vec<usize>,
        row_idx: Vec<usize>,
        val: Vec<f64>,
    },
    Scalar {
        cols: Vec<usize>,
        inv: Vec<f64>,
    },
}

impl std::fmt::Debug for Ic0Factor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ic0Factor::IncompleteChol { cols, val, .. } => write!(
                f,
                "Ic0Factor::IncompleteChol {{ cols.len: {}, nnz: {} }}",
                cols.len(),
                val.len()
            ),
            Ic0Factor::Scalar { cols, .. } => {
                write!(f, "Ic0Factor::Scalar {{ cols.len: {} }}", cols.len())
            }
        }
    }
}

/// Level-0 incomplete-Cholesky Schur preconditioner (#299).
///
/// One sparse incomplete-Cholesky factor per connected component of the
/// β-coupling graph. Within a component the dense `S[C,C]` is assembled, its
/// structural-nonzero pattern `P = { (i,j) : |S_ij| > drop·sqrt(S_ii S_jj) }`
/// is taken as the level-0 fill set, and the no-fill incomplete Cholesky
/// `S ≈ L̃ L̃ᵀ` is formed keeping only `P` (drop any update landing outside it).
#[derive(Debug, Clone)]
pub(crate) struct BlockIncompleteCholeskyPreconditioner {
    pub(crate) components: Vec<Ic0Factor>,
}

impl BlockIncompleteCholeskyPreconditioner {
    pub fn from_arrow_schur<B: BatchedBlockSolver + Sync>(
        sys: &ArrowSchurSystem,
        htt_factors: &ArrowFactorSlab,
        ridge_beta: f64,
        backend: &B,
    ) -> Result<Self, ArrowSchurError> {
        // Column grouping mirrors ClusterJacobi: one group per connected
        // component of the β-coupling graph (whole-K single group when no
        // block_offsets are registered), so IC(0) preconditions exactly the
        // coupling ClusterJacobi keeps, but with a sparse (no-fill) factor.
        let col_groups: Vec<Vec<usize>> = if sys.block_offsets.is_empty() {
            vec![(0..sys.k).collect()]
        } else {
            let graph = BetaCouplingGraph::build_from_system(sys);
            graph
                .component_partition()
                .iter()
                .map(|comp| {
                    let mut cols: Vec<usize> = comp
                        .iter()
                        .flat_map(|&blk| sys.block_offsets[blk].clone())
                        .collect();
                    cols.sort_unstable();
                    cols.dedup();
                    cols
                })
                .collect()
        };

        let mut components = Vec::with_capacity(col_groups.len());
        for cols in &col_groups {
            let b = cols.len();
            if b == 0 {
                continue;
            }
            if b > IC0_MAX_COMPONENT {
                let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                components.push(Ic0Factor::Scalar {
                    cols: cols.clone(),
                    inv,
                });
                continue;
            }
            let mut s_block =
                assemble_local_schur_block(sys, htt_factors, ridge_beta, backend, cols);
            symmetrize_upper_from_lower(&mut s_block);
            match incomplete_cholesky_level0(&s_block) {
                Some((col_ptr, row_idx, val)) => components.push(Ic0Factor::IncompleteChol {
                    cols: cols.clone(),
                    col_ptr,
                    row_idx,
                    val,
                }),
                None => {
                    // Non-PD incomplete pivot: degrade this component to the
                    // scalar reciprocal diagonal (mirrors the ClusterJacobi
                    // non-PD fallback), which is always applicable for a
                    // PD-floored Schur diagonal.
                    let inv = build_schur_scalar_inv(sys, htt_factors, ridge_beta, backend, cols)?;
                    components.push(Ic0Factor::Scalar {
                        cols: cols.clone(),
                        inv,
                    });
                }
            }
        }
        Ok(Self { components })
    }

    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(r.len());
        for comp in &self.components {
            match comp {
                Ic0Factor::Scalar { cols, inv } => {
                    for (local, &gi) in cols.iter().enumerate() {
                        out[gi] = inv[local] * r[gi];
                    }
                }
                Ic0Factor::IncompleteChol {
                    cols,
                    col_ptr,
                    row_idx,
                    val,
                } => {
                    let b = cols.len();
                    // Gather the local residual, solve `L̃ L̃ᵀ z = r_local` by a
                    // sparse forward solve (`L̃ y = r`) then a sparse back solve
                    // (`L̃ᵀ z = y`), then scatter `z` back to global columns.
                    let mut z = vec![0.0f64; b];
                    for (local, &gi) in cols.iter().enumerate() {
                        z[local] = r[gi];
                    }
                    // Forward solve `L̃ y = r` (overwrite z with y). Column-major
                    // CSC: row_idx[col_ptr[j]] == j (diagonal stored first).
                    for j in 0..b {
                        let dstart = col_ptr[j];
                        let diag = val[dstart];
                        z[j] /= diag;
                        let yj = z[j];
                        for k in (dstart + 1)..col_ptr[j + 1] {
                            z[row_idx[k]] -= val[k] * yj;
                        }
                    }
                    // Back solve `L̃ᵀ z = y` (overwrite z). Walk columns in
                    // reverse; the below-diagonal entries of column j are the
                    // off-diagonal entries of row j of L̃ᵀ.
                    for j in (0..b).rev() {
                        let dstart = col_ptr[j];
                        let mut acc = z[j];
                        for k in (dstart + 1)..col_ptr[j + 1] {
                            acc -= val[k] * z[row_idx[k]];
                        }
                        z[j] = acc / val[dstart];
                    }
                    for (local, &gi) in cols.iter().enumerate() {
                        out[gi] = z[local];
                    }
                }
            }
        }
        out
    }
}

/// Level-0 incomplete Cholesky of a dense SPD-ish block `a` (`b×b`, symmetric).
///
/// Returns the lower factor `L̃` in column-compressed (CSC) form
/// `(col_ptr, row_idx, val)` where each column lists its diagonal entry FIRST
/// followed by the strictly-below-diagonal entries, in increasing row order.
/// The kept pattern is the level-0 set `P` = structural nonzeros of `a` (a
/// relative drop threshold prunes round-off). IC(0) computes the standard
/// Cholesky recurrence but DROPS any value at a position outside `P`, so the
/// factor has exactly `nnz(tril(P))` entries — no fill. Returns `None` on a
/// non-positive pivot (caller degrades to scalar diagonal).
///
/// Reference: Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed.,
/// §10.3.2 (IC(0)). This is the left-looking, pattern-restricted variant.
pub(crate) fn incomplete_cholesky_level0(
    a: &Array2<f64>,
) -> Option<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let b = a.nrows();
    assert_eq!(a.ncols(), b, "incomplete Cholesky needs a square block");

    // ---- derive the level-0 lower-triangular pattern from `a` --------------
    // Per column j, the kept below-or-on-diagonal rows i>=j with a structurally
    // nonzero a[i,j]. The diagonal is always kept.
    let mut col_ptr = vec![0usize; b + 1];
    let mut row_idx: Vec<usize> = Vec::new();
    // value buffer, parallel to row_idx, initialised from tril(a) on the pattern
    let mut val: Vec<f64> = Vec::new();
    // For O(1) "is (i,j) in pattern + where" lookups during the recurrence, keep
    // a per-column map from global row -> position in that column's value slice.
    let mut col_pos: Vec<std::collections::HashMap<usize, usize>> = Vec::with_capacity(b);
    for j in 0..b {
        let ajj = a[[j, j]];
        let scale_j = ajj.abs().max(0.0).sqrt();
        let mut map = std::collections::HashMap::new();
        // diagonal first
        map.insert(j, val.len());
        row_idx.push(j);
        val.push(ajj);
        for i in (j + 1)..b {
            let aij = a[[i, j]];
            let scale_i = a[[i, i]].abs().sqrt();
            let thresh = IC0_PATTERN_REL_DROP * scale_i * scale_j;
            if aij.abs() > thresh {
                map.insert(i, val.len());
                row_idx.push(i);
                val.push(aij);
            }
        }
        col_pos.push(map);
        col_ptr[j + 1] = val.len();
    }

    // ---- IC(0) recurrence, left-looking over columns -----------------------
    // For column j: subtract the contributions of all prior columns k<j that
    // have BOTH a nonzero at row j (so they touch the diagonal/the column) — the
    // multiplier L[j,k] — and a nonzero at the rows i of column j's pattern.
    // Any update whose target (i,j) is OUTSIDE the kept pattern is dropped.
    for j in 0..b {
        // Diagonal: a[j,j] - Σ_{k<j} L[j,k]². Each prior column k<j contributes
        // its row-j entry L[j,k] (looked up by row, so the column index is not
        // needed); columns without a row-j entry contribute nothing.
        let dpos = col_ptr[j];
        let mut diag = val[dpos];
        for mapk in &col_pos[..j] {
            if let Some(&pjk) = mapk.get(&j) {
                let ljk = val[pjk];
                diag -= ljk * ljk;
            }
        }
        if !diag.is_finite() || diag <= 0.0 {
            return None;
        }
        let ljj = diag.sqrt();
        val[dpos] = ljj;
        // Below-diagonal of column j: L[i,j] = (a[i,j] - Σ_{k<j} L[i,k] L[j,k]) / L[j,j]
        for p in (dpos + 1)..col_ptr[j + 1] {
            let i = row_idx[p];
            let mut s = val[p];
            for mapk in &col_pos[..j] {
                if let (Some(&pik), Some(&pjk)) = (mapk.get(&i), mapk.get(&j)) {
                    s -= val[pik] * val[pjk];
                }
            }
            val[p] = s / ljj;
        }
    }
    Some((col_ptr, row_idx, val))
}

/// Build scalar diagonal inverses for a set of global column indices.
///
/// Used when a cluster is non-PD or exceeds `CLUSTER_JACOBI_MAX_CLUSTER`.
pub(crate) fn build_schur_scalar_inv<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    backend: &B,
    cols: &[usize],
) -> Result<Vec<f64>, ArrowSchurError> {
    let mut result = Vec::with_capacity(cols.len());
    // Extract the penalty diagonal for all K columns once, then index per-column.
    let mut full_diag = Array1::<f64>::zeros(sys.k);
    {
        let diag_slice = full_diag.as_slice_mut().expect("full_diag contiguous");
        sys.penalty_diagonal_add(diag_slice);
    }
    // Probe each needed column through the ROUTED `H_tβ` convention at each
    // row's own width (see `assemble_local_schur_block` for why a raw
    // `row.htbeta` read at the global `sys.d` is wrong here).
    let mut e_g = Array1::<f64>::zeros(sys.k);
    for &gi in cols {
        let mut s = full_diag[gi] + ridge_beta;
        e_g[gi] = 1.0;
        for (row_idx, row) in sys.rows.iter().enumerate() {
            let di = sys.row_dims[row_idx];
            let mut col_vec = Array1::<f64>::zeros(di);
            sys_htbeta_apply_row(sys, row_idx, row, e_g.view(), &mut col_vec);
            let solved = backend.solve_block_vector(htt_factors.factor(row_idx), col_vec.view());
            let mut acc = 0.0;
            for c in 0..di {
                acc += col_vec[c] * solved[c];
            }
            s -= acc;
        }
        e_g[gi] = 0.0;
        if !s.is_finite() || s <= 0.0 {
            return Err(ArrowSchurError::PcgFailed {
                reason: format!(
                    "cluster Schur scalar fallback: non-PD diagonal at index {gi}: {s}"
                ),
            });
        }
        result.push(1.0 / s);
    }
    Ok(result)
}

/// Inexact PCG with automatic preconditioner-ladder escalation.
///
/// Starts with `JacobiPreconditioner` (Diagonal or BetaBlockJacobi).
/// If PCG spends its resolved `budget` without converging
/// (`BudgetExhausted`) and `k > PRECOND_ESCALATE_K_THRESHOLD`, escalates to
/// `ClusterJacobi`; if that also spends its budget, escalates to
/// `AdditiveSchwarz { overlap: 1 }`. Each tier may spend the same budget.
pub(crate) fn steihaug_pcg_auto<B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    rhs: &Array1<f64>,
    budget: ArrowPcgBudget,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    backend: &B,
    gpu_matvec: Option<&GpuSchurMatvec>,
    curvature_floor: Option<f64>,
) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurError> {
    // #1017 CPU residency: stage the per-row reduced-Schur factors `(L_i, Y_i)`
    // (NOT the dense `p×p` block — `di ≪ p`, so the factored form is `O(n·di·p)`
    // memory and `2·support_i·p + 2·di·p` flops/row including the sparse
    // gather/scatter over the active support) once, up
    // front, when the SAE structure is installed and the matvec runs on host
    // (CPU). The GPU matvec carries its own residency, so skip when it is engaged.
    // The same staged operator is reused across the whole preconditioner ladder
    // (Jacobi → ClusterJacobi → AdditiveSchwarz) — built once, not per tier.
    let resident = if gpu_matvec.is_none() {
        SaeResidentReducedSchur::build(sys, htt_factors, backend)
    } else {
        None
    };
    // #2228 — a β-gauge-quotiented system has a reduced Schur that is singular
    // along the gauge orbit, and every preconditioner in the ladder below
    // (block-Jacobi, cluster, Schwarz, IC(0)) is formed from the UN-pinned
    // operator, so it would misprice — or refuse as non-PD — that orbit
    // direction. The matvec now applies the Faddeev–Popov pin `P S P + Q Qᵀ`,
    // which is SPD and well-conditioned on the identifiable complement (the gauge
    // dimension is tiny — one direction per circle/torus phase), so an identity
    // preconditioner converges without a bespoke pinned diagonal. Route straight
    // through it and skip the diagonal ladder, whose preconditioners assume the
    // un-pinned Schur; the `None`-quotient path below is byte-identical.
    if sys.beta_gauge_quotient.is_some() {
        let identity = IdentityPreconditioner;
        let (step, diag) = run_pcg_with_preconditioner(
            sys,
            htt_factors,
            ridge_beta,
            rhs,
            |r| identity.apply(r),
            budget,
            pcg,
            trust,
            backend,
            gpu_matvec,
            resident.as_ref(),
        )?;
        // Mirror the non-gauge contract: below the escalation threshold a
        // `BudgetExhausted` stop is accepted (the ladder returns it as `Ok`); above it
        // the ladder would escalate the preconditioner, but the cluster/Schwarz/IC(0)
        // tiers assume the un-pinned Schur and cannot precondition the gauge pin, so
        // surface the typed budget refusal and let the outer LM loop escalate the
        // ridge instead (a bespoke pinned-diagonal preconditioner is the follow-up).
        if diag.stopping_reason == PcgStopReason::BudgetExhausted
            && sys.k > PRECOND_ESCALATE_K_THRESHOLD
        {
            return Err(ArrowSchurError::PcgBudgetExhausted {
                budget,
                products_spent: diag.matvec_calls,
                final_relative_residual: diag.final_relative_residual,
            });
        }
        return Ok((step, diag));
    }
    // #1026 — curvature-floor retry on the Jacobi tier. The unbounded SAE inner
    // PCG (trust radius = ∞) fails on `pᵀSp ≤ 0` when the reduced Schur is
    // indefinite (K≥4 co-collapse: a near-singular per-row `H_tt` over-subtracts
    // `S`). Instead of letting that failure propagate to the outer LM loop —
    // which inflates `ridge_β` over EVERY β direction and makes the inner Newton
    // crawl — floor the OPERATOR by the minimal ridge `δ = |pᵀSp|/‖p‖² · (1+ε)`
    // that restores positive curvature along the offending direction, rebuild the
    // Jacobi preconditioner at the lifted ridge, and retry. This is the
    // matrix-free analogue of the dense `spectral_pd_floored_schur`: the healthy
    // β subspace (where curvature is already positive) is essentially untouched
    // by a tiny `δ`, while the collapsed direction gets exactly the stiffness it
    // needs to make a real descent step. A PD reduced Schur never hits `pᵀSp ≤ 0`,
    // so this loop is a strict no-op there (bit-for-bit unchanged). Bounded by a
    // small attempt cap and a relative ridge ceiling; on exhaustion the original
    // recoverable failure still reaches the outer LM loop.
    let mut effective_ridge = ridge_beta;
    let mut x0_diag0: Option<(Array1<f64>, ArrowPcgDiagnostics)> = None;
    let mut last_curvature_err: Option<ArrowSchurError> = None;
    let rhs_scale = euclidean_norm(rhs.view()).max(1.0);
    let ridge_ceiling = ridge_beta.max(SCHUR_CURVATURE_FLOOR_REL_CEILING * rhs_scale);
    for _attempt in 0..=SCHUR_CURVATURE_FLOOR_MAX_ATTEMPTS {
        // The Jacobi preconditioner build itself refuses a non-PD Schur diagonal
        // (`PcgFailed: invalid Schur Jacobi diagonal`) — the SAME co-collapse
        // signature reached BEFORE the CG loop, since `S_ii = H_ββ,ii − Σ …` goes
        // negative. Treat that build failure as a curvature deficit too: when the
        // floor is enabled, lift the ridge and retry; otherwise propagate.
        let jacobi = match JacobiPreconditioner::from_arrow_schur(
            sys,
            htt_factors,
            effective_ridge,
            backend,
            resident.as_ref(),
        ) {
            Ok(jacobi) => jacobi,
            Err(err @ ArrowSchurError::PcgFailed { .. }) => {
                if curvature_floor.is_none() {
                    return Err(err);
                }
                // A diagonal refusal carries no `(curvature, ‖p‖²)` deficit, and
                // the over-subtraction magnitude `Σ H_tβᵀ(H_tt)⁻¹H_tβ` is
                // unbounded relative to `rhs_scale`, so a small additive bump
                // would crawl. Escalate the ridge MULTIPLICATIVELY (×10, matching
                // the per-row `factor_one_row_result` RIDGE_GROWTH_FACTOR), seeded
                // at `rhs_scale`, so even a large deficit (the collapsed
                // `(H_tβ)²/H_tt` over-subtraction) is reached in a handful of
                // attempts. The ceiling + attempt cap still bound it; on
                // exhaustion the recoverable failure reaches the outer LM loop.
                // Jump straight to a meaningful scale on the FIRST refusal rather
                // than crawling ×10 from a tiny `ridge_beta`: each rebuild is a full
                // block-Jacobi factorization (the massive-K preconditioner hotspot),
                // and a large collapsed deficit (`Σ H_tβᵀ(H_tt)⁻¹H_tβ` over-subtraction,
                // O(1)-scale) otherwise costs ~log10(deficit / ridge_beta) rebuilds.
                // Seeding the first bump at `rhs_scale` covers it in one or two, then
                // escalates multiplicatively; the ceiling + attempt cap still bound it.
                let next = if effective_ridge > 0.0 {
                    (effective_ridge * SCHUR_CURVATURE_FLOOR_DIAG_GROWTH).max(rhs_scale)
                } else {
                    rhs_scale
                };
                last_curvature_err = Some(err);
                if !next.is_finite() || next > ridge_ceiling {
                    break;
                }
                effective_ridge = next;
                continue;
            }
            Err(other) => return Err(other),
        };
        match run_pcg_with_preconditioner(
            sys,
            htt_factors,
            effective_ridge,
            rhs,
            |r| jacobi.apply(r),
            budget,
            pcg,
            trust,
            backend,
            gpu_matvec,
            resident.as_ref(),
        ) {
            Ok(result) => {
                x0_diag0 = Some(result);
                break;
            }
            Err(ArrowSchurError::UnboundedNegativeCurvature {
                curvature,
                direction_norm_sq,
            }) => {
                // Only floor when the caller opted in (SAE solve path); otherwise
                // propagate the raw negative-curvature signal so BA / non-SAE
                // unbounded solves keep their existing failure contract.
                let Some(relative_floor) = curvature_floor else {
                    return Err(ArrowSchurError::UnboundedNegativeCurvature {
                        curvature,
                        direction_norm_sq,
                    });
                };
                // Minimal ridge to make `pᵀ(S+δI)p = |curvature| + δ·‖p‖² > 0`,
                // with a margin so the next CG iterate has strictly positive
                // curvature rather than sitting on the `0` knife-edge.
                let deficit = if direction_norm_sq > 0.0 {
                    curvature.abs() / direction_norm_sq
                } else {
                    0.0
                };
                let bump = (deficit * (1.0 + SCHUR_CURVATURE_FLOOR_MARGIN))
                    .max(relative_floor.max(SCHUR_CURVATURE_FLOOR_REL_FLOOR) * rhs_scale);
                let next = (effective_ridge + bump).max(effective_ridge * 2.0);
                last_curvature_err = Some(ArrowSchurError::UnboundedNegativeCurvature {
                    curvature,
                    direction_norm_sq,
                });
                if !next.is_finite() || next > ridge_ceiling {
                    break;
                }
                effective_ridge = next;
            }
            Err(other) => return Err(other),
        }
    }
    let (x0, diag0) = match x0_diag0 {
        Some(result) => result,
        None => {
            // The curvature floor could not condition the operator within the
            // ceiling; hand the recoverable failure to the outer LM loop, which
            // re-forms the system at a heavier ridge.
            return Err(last_curvature_err.unwrap_or(ArrowSchurError::PcgFailed {
                reason: "unbounded Schur PCG negative curvature unresolved by curvature floor"
                    .to_string(),
            }));
        }
    };
    if sys.k <= PRECOND_ESCALATE_K_THRESHOLD
        || diag0.stopping_reason != PcgStopReason::BudgetExhausted
    {
        return Ok((x0, diag0));
    }
    // Escalation tiers reuse the curvature-floored `effective_ridge` so the
    // operator they precondition is the SAME (PD-floored) one the Jacobi tier
    // settled on; a still-negative-curvature signal here is handed to the outer
    // LM loop (it only arises if the floored Jacobi tier merely ran out of
    // iterations yet a coarser preconditioner still finds an indefinite
    // direction — rare; the LM loop re-forms at a heavier ridge).
    // Default cluster tier: the bounded CO-VISIBILITY partition, not the
    // connected-component partition. At the SAE widths this ladder targets the
    // co-firing graph is one giant component, so the component partition exceeds
    // the size cap and `from_arrow_schur` degrades to scalar Jacobi (the ceiling
    // this tier exists to lift). `from_arrow_schur_covisibility` splits that
    // component into bounded strongly-co-firing clusters whose dense factors
    // condition the cross-atom coupling scalar Jacobi drops. The component
    // partition stays selectable via `from_arrow_schur` (used by the ladder
    // study and its regression gates). Both precondition the SAME operator, so
    // the converged step — and the REML optimum — is unchanged.
    let cluster = ClusterJacobiPreconditioner::from_arrow_schur_covisibility(
        sys,
        htt_factors,
        effective_ridge,
        backend,
    )?;
    let (x1, diag1) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        effective_ridge,
        rhs,
        |r| cluster.apply(r),
        budget,
        pcg,
        trust,
        backend,
        gpu_matvec,
        resident.as_ref(),
    )?;
    if diag1.stopping_reason != PcgStopReason::BudgetExhausted {
        return Ok((x1, diag1));
    }
    let schwarz = AdditiveSchwarzPreconditioner::from_arrow_schur(
        sys,
        htt_factors,
        effective_ridge,
        backend,
        1,
    )?;
    let (x2, diag2) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        effective_ridge,
        rhs,
        |r| schwarz.apply(r),
        budget,
        pcg,
        trust,
        backend,
        gpu_matvec,
        resident.as_ref(),
    )?;
    if diag2.stopping_reason != PcgStopReason::BudgetExhausted {
        return Ok((x2, diag2));
    }
    // Final tier — diagonal-assembled additive Schwarz (#299), the cheap-apply
    // Schwarz variant. When the dense-block AdditiveSchwarz still ran out of
    // iterations its O(Σ b_k²) apply may have throttled the iteration budget on
    // a wide subdomain; the diag-assembled variant keeps Schwarz's overlapping
    // local-inverse conditioning but applies in O(K), so it can take more CG
    // iterations within the same wall budget. Same overlap (1) and same
    // curvature-floored ridge as the dense-block tier.
    let diag_schwarz = DiagAssembledSchwarzPreconditioner::from_arrow_schur(
        sys,
        htt_factors,
        effective_ridge,
        backend,
        1,
    )?;
    let (x3, diag3) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        effective_ridge,
        rhs,
        |r| diag_schwarz.apply(r),
        budget,
        pcg,
        trust,
        backend,
        gpu_matvec,
        resident.as_ref(),
    )?;
    if diag3.stopping_reason != PcgStopReason::BudgetExhausted {
        return Ok((x3, diag3));
    }
    // Richest tier — level-0 incomplete Cholesky (#299). ClusterJacobi keeps the
    // full DENSE Cholesky of each component (so on a single large connected
    // component it fills the whole `b×b` factor and its `O(b²)` apply throttles
    // the CG iteration budget), while the diagonal/Schwarz tiers drop most
    // inter-block coupling. IC(0) keeps the component's full structural coupling
    // but only the level-0 (no-fill) pattern, so its sparse triangular apply is
    // `O(nnz(S[C,C]))` — it can take more CG iterations within the same wall
    // budget AND conditions the off-diagonal coupling the cheap tiers discard.
    // Last in the ladder so it is only paid when every cheaper tier stalled.
    let ic0 = BlockIncompleteCholeskyPreconditioner::from_arrow_schur(
        sys,
        htt_factors,
        effective_ridge,
        backend,
    )?;
    let (x4, diag4) = run_pcg_with_preconditioner(
        sys,
        htt_factors,
        effective_ridge,
        rhs,
        |r| ic0.apply(r),
        budget,
        pcg,
        trust,
        backend,
        gpu_matvec,
        resident.as_ref(),
    )?;
    // All five preconditioner tiers (Jacobi -> ClusterJacobi -> AdditiveSchwarz
    // -> DiagAssembledSchwarz -> BlockIncompleteCholesky) spent their product
    // budget without driving the residual below tolerance. Returning a
    // truncated iterate as `Ok` would feed an arbitrarily-large-residual step
    // into the Newton driver, where the PCG diagnostics are discarded. Surface the
    // typed budget refusal instead so `solve_with_lm_escalation_inner` escalates
    // the proximal ridge: better conditioning is precisely what a stalled PCG on
    // an ill-conditioned reduced system needs.
    if diag4.stopping_reason == PcgStopReason::BudgetExhausted {
        return Err(ArrowSchurError::PcgBudgetExhausted {
            budget,
            products_spent: [&diag0, &diag1, &diag2, &diag3, &diag4]
                .iter()
                .map(|diag| diag.matvec_calls)
                .sum(),
            final_relative_residual: diag4.final_relative_residual,
        });
    }
    Ok((x4, diag4))
}

/// Run Steihaug-CG with a generic preconditioner closure.
/// Routes matvec through GPU when `gpu_matvec` is set.
pub(crate) fn run_pcg_with_preconditioner<ApplyPrec, B: BatchedBlockSolver + Sync>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    rhs: &Array1<f64>,
    apply_prec: ApplyPrec,
    budget: ArrowPcgBudget,
    pcg: &ArrowPcgOptions,
    trust: &ArrowTrustRegionOptions,
    backend: &B,
    gpu_matvec: Option<&GpuSchurMatvec>,
    resident: Option<&SaeResidentReducedSchur>,
) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurError>
where
    ApplyPrec: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let tol = pcg
        .relative_tolerance
        .max(trust.steihaug_relative_tolerance);
    // #2228 — route the fit-step matvec through `ReducedSchurOperator`, which
    // applies the Faddeev–Popov pin `v ↦ P S P v + Q Qᵀ v` when the system carries
    // a β-gauge quotient and is byte-for-byte the bare `gpu_matvec` / `schur_matvec`
    // apply when it does not. This gauge-fixes the wide-`p` InexactPCG Newton step
    // exactly like the dense Direct/SqrtBA modes while leaving the `None`-quotient
    // lane (every non-SAE-fit caller) unchanged.
    let op = ReducedSchurOperator::new(sys, htt_factors, ridge_beta, backend, resident)
        .with_gpu_matvec(gpu_matvec);
    steihaug_cg(
        rhs,
        |p, out| op.apply_into(p, out),
        apply_prec,
        budget,
        tol,
        trust.radius,
    )
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct IdentityPreconditioner;

impl IdentityPreconditioner {
    pub(crate) fn apply(&self, r: &Array1<f64>) -> Array1<f64> {
        r.clone()
    }
}

/// Steihaug-CG on an assembled dense Schur, the trust-region correction of a dense
/// step that left the trust ball. The Schur is already factored, so CG may take the
/// Krylov dimension of the system (#2900 row 6.15).
pub(crate) fn steihaug_dense_system(
    schur: &Array2<f64>,
    rhs: &Array1<f64>,
    preconditioner: &IdentityPreconditioner,
    trust: &ArrowTrustRegionOptions,
) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurError> {
    steihaug_cg(
        rhs,
        |p, out| dense_matvec(schur, p, out),
        |r| preconditioner.apply(r),
        ArrowPcgBudget::krylov_dimension(schur.nrows()),
        trust.steihaug_relative_tolerance,
        trust.radius,
    )
}

/// Steihaug-CG on `matvec` within the resolved product `budget` (#2900 row 6.15).
///
/// The only budget stop is [`ArrowPcgBudget::stop_at`], read before each product and
/// after the accuracy test, so a solve that meets its tolerance on the last product
/// it may spend succeeds. A spent budget returns the truncated iterate with
/// [`PcgStopReason::BudgetExhausted`].
pub(crate) fn steihaug_cg<MatVec, ApplyPrec>(
    rhs: &Array1<f64>,
    mut matvec: MatVec,
    mut apply_preconditioner: ApplyPrec,
    budget: ArrowPcgBudget,
    relative_tolerance: f64,
    trust_radius: f64,
) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurError>
where
    MatVec: FnMut(&Array1<f64>, &mut Array1<f64>),
    ApplyPrec: FnMut(&Array1<f64>) -> Array1<f64>,
{
    let n = rhs.len();
    let radius = if trust_radius.is_finite() && trust_radius > 0.0 {
        trust_radius
    } else {
        f64::INFINITY
    };
    let rhs_norm = euclidean_norm(rhs.view());
    if rhs_norm == 0.0 {
        return Ok((Array1::<f64>::zeros(n), ArrowPcgDiagnostics::default()));
    }
    let tol = (relative_tolerance.max(0.0) * rhs_norm).max(PCG_ABSOLUTE_TOLERANCE_FLOOR);
    let mut x = Array1::<f64>::zeros(n);
    let mut r = rhs.clone();
    let mut z = apply_preconditioner(&r);
    let mut diag = ArrowPcgDiagnostics {
        precond_apply_calls: 1,
        ..ArrowPcgDiagnostics::default()
    };
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    if rz <= 0.0 || !rz.is_finite() {
        if radius.is_finite() {
            diag.final_relative_residual = euclidean_norm(r.view()) / rhs_norm;
            diag.stopping_reason = PcgStopReason::TrustRegion;
            return Ok((step_to_trust_boundary(&x, &r, radius), diag));
        }
        // Unbounded (radius = ∞) non-positive preconditioned residual: the
        // reduced Schur is indefinite at the very first direction. Surface the
        // typed curvature-floor signal so `steihaug_pcg_auto` floors the
        // operator minimally and retries, instead of failing into a global
        // `ridge_β` ramp. `rz = rᵀM⁻¹r` is a preconditioner-metric curvature;
        // report it with the residual norm² as the direction scale.
        return Err(ArrowSchurError::UnboundedNegativeCurvature {
            curvature: rz,
            direction_norm_sq: dot(&r, &r),
        });
    }
    if euclidean_norm(r.view()) <= tol {
        diag.final_relative_residual = 0.0;
        diag.stopping_reason = PcgStopReason::Converged;
        return Ok((x, diag));
    }
    let mut ap = Array1::<f64>::zeros(n);
    // Reused candidate scratch — avoid per-iteration clone of x.
    let mut candidate = Array1::<f64>::zeros(n);
    loop {
        if let Some(reason) = budget.stop_at(diag.matvec_calls) {
            diag.final_relative_residual = euclidean_norm(r.view()) / rhs_norm;
            diag.stopping_reason = reason;
            return Ok((x, diag));
        }
        matvec(&p, &mut ap);
        diag.matvec_calls += 1;
        diag.iterations += 1;
        let pap = dot(&p, &ap);
        if pap <= 0.0 || !pap.is_finite() {
            if radius.is_finite() {
                diag.final_relative_residual = euclidean_norm(r.view()) / rhs_norm;
                diag.stopping_reason = PcgStopReason::TrustRegion;
                return Ok((step_to_trust_boundary(&x, &p, radius), diag));
            }
            // Unbounded negative curvature `pᵀSp ≤ 0`: the reduced Schur is
            // indefinite along `p` (the #1026 co-collapse direction). Surface
            // the typed signal carrying `pᵀSp` and `‖p‖²` so the caller floors
            // the operator by the minimal ridge `δ = |pᵀSp|/‖p‖²` (which makes
            // `pᵀ(S+δI)p = 0⁺`) plus a margin, and retries.
            return Err(ArrowSchurError::UnboundedNegativeCurvature {
                curvature: pap,
                direction_norm_sq: dot(&p, &p),
            });
        }
        let alpha = rz / pap;
        for i in 0..n {
            candidate[i] = x[i] + alpha * p[i];
        }
        if radius.is_finite() && euclidean_norm(candidate.view()) >= radius {
            diag.final_relative_residual = euclidean_norm(r.view()) / rhs_norm;
            diag.stopping_reason = PcgStopReason::TrustRegion;
            return Ok((step_to_trust_boundary(&x, &p, radius), diag));
        }
        x.assign(&candidate);
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }
        if euclidean_norm(r.view()) <= tol {
            diag.final_relative_residual = euclidean_norm(r.view()) / rhs_norm;
            diag.stopping_reason = PcgStopReason::Converged;
            return Ok((x, diag));
        }
        z = apply_preconditioner(&r);
        diag.precond_apply_calls += 1;
        let rz_next = dot(&r, &z);
        if rz_next <= 0.0 || !rz_next.is_finite() {
            return Err(ArrowSchurError::PcgFailed {
                reason: "non-positive or non-finite PCG residual".to_string(),
            });
        }
        let beta = rz_next / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_next;
    }
}

pub(crate) fn step_to_trust_boundary(
    x: &Array1<f64>,
    p: &Array1<f64>,
    radius: f64,
) -> Array1<f64> {
    let pp = dot(p, p);
    if pp == 0.0 {
        return x.clone();
    }
    let xp = dot(x, p);
    let xx = dot(x, x);
    let disc = (xp * xp + pp * (radius * radius - xx)).max(0.0);
    let tau = (-xp + disc.sqrt()) / pp;
    let mut out = x.clone();
    for i in 0..out.len() {
        out[i] += tau * p[i];
    }
    out
}

pub(crate) fn dense_matvec(a: &Array2<f64>, x: &Array1<f64>, out: &mut Array1<f64>) {
    let n = a.nrows();
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += a[[i, j]] * x[j];
        }
        out[i] = acc;
    }
}

pub(crate) fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

pub(crate) fn euclidean_norm(v: ArrayView1<'_, f64>) -> f64 {
    let mut acc = 0.0;
    for x in v.iter() {
        acc += x * x;
    }
    acc.sqrt()
}

pub(crate) fn symmetrize_upper_from_lower(a: &mut Array2<f64>) {
    let n = a.nrows().min(a.ncols());
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = v;
            a[[j, i]] = v;
        }
    }
}

/// Errors raised by [`ArrowSchurSystem::solve`].
#[derive(Debug, Clone)]
pub enum ArrowSchurError {
    /// A per-row `H_tt^(i)` block was not positive-definite at the
    /// supplied ridge. Indicates an under-regularized latent block —
    /// typically a gauge-free fit without an identifiability penalty.
    PerRowFactorFailed { row: usize, reason: String },
    /// A per-row `H_tt^(i)` block factored, but the Cholesky factor failed
    /// the safe-inversion guard for the Schur reduction. This can be either
    /// an excessive diagonal-ratio condition-number estimate or a numerically
    /// tiny pivot relative to the row block scale. Cholesky technically
    /// succeeded, but the inverse used in
    /// `S = H_ββ − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)` is contaminated
    /// by spectral terms on the order of `κ_i`; functionally
    /// equivalent to a PSD-fail for Schur stability. The LM outer
    /// wrapper escalates `ridge_t` identically to `PerRowFactorFailed`.
    PerRowFactorIllConditioned { row: usize, kappa_estimate: f64 },
    /// The Schur complement was not positive-definite. Indicates a
    /// near-collinear decoder or a degenerate weighting; the LM outer
    /// wrapper should escalate `ridge_beta` and retry.
    SchurFactorFailed { reason: String },
    /// The BA inexact-step PCG solve failed before producing a usable
    /// Steihaug trust-region step.
    PcgFailed { reason: String },
    /// The inexact PCG spent its resolved product budget without meeting its
    /// forcing tolerance, and nothing answers the miss (#2900 row 6.15): the
    /// request asked for InexactPCG, or no dense route fits the materialization
    /// cap. `products_spent` counts the products of every launch the refusal
    /// covers, and `final_relative_residual` is the recursive `‖r̂‖/‖rhs‖` at the
    /// last one. Floating-point CG can legitimately need more products than the
    /// Krylov dimension, so the residual tells a solve cut off while still
    /// converging from one that stalled.
    PcgBudgetExhausted {
        budget: ArrowPcgBudget,
        products_spent: usize,
        final_relative_residual: f64,
    },
    /// The UNBOUNDED (trust-radius = ∞) Schur PCG encountered negative
    /// curvature `pᵀSp ≤ 0` (or a non-positive preconditioned residual): the
    /// reduced Schur is indefinite, the #1026 K≥4 co-collapse signature where
    /// a near-singular per-row `H_tt` over-subtracts `S`. With no trust radius
    /// there is no boundary to step to, so CG cannot proceed. `curvature` is
    /// the offending `pᵀSp` and `direction_norm_sq` the `‖p‖²` of the
    /// negative-curvature direction; the caller floors the operator with the
    /// minimal ridge `δ = (|curvature|/‖p‖² )·(1+ε)` that restores positive
    /// curvature along `p` and retries (matrix-free analogue of the dense
    /// `spectral_pd_floored_schur`), rather than blindly inflating `ridge_β`.
    UnboundedNegativeCurvature {
        curvature: f64,
        direction_norm_sq: f64,
    },
    /// Adaptive proximal damping could not produce an Armijo-accepted
    /// nonlinear step.
    AdaptiveCorrectionFailed { reason: String },
    /// The proximal ridge ladder refused at a rung the system's declared bounds
    /// certify factorable (#2627): every factorization guard provably passes
    /// there, so no larger shift can cure the refusal in `cause`.
    RefusedAtCertifiedShift {
        proximal_ridge: f64,
        cause: Box<ArrowSchurError>,
    },
}

impl ArrowSchurError {
    /// Whether a rendered [`ArrowSchurError`] message reports an indefinite reduced
    /// Schur complement: a `SchurFactorFailed` whose reason names a non-PD operator.
    /// A `SchurFactorFailed` for a non-finite entry, a non-square operator or an
    /// unavailable device, and every other variant, stays fatal.
    ///
    /// #2598 — gam-sae's ρ-probe classifier is one such caller: by the time a
    /// refusal reaches `ProbeRefusalKind::classify` the spine has flattened it
    /// to a `String`, and it was recovering this verdict by matching two
    /// literals of the [`Display`] impl below — in another crate. That made
    /// **rewording either message here a silent reclassification of every
    /// recoverable Schur refusal as a fatal defect**, with nothing failing.
    ///
    /// The wording knowledge now lives beside the wording. The discriminant
    /// phrase and the reason phrase are the value-level conjunct, and
    /// `rendered_verdict_matches_the_value_verdict_for_every_variant_2598` pins
    /// this reader to that verdict for every variant, so a reword must move both
    /// together or fail there.
    ///
    /// `contains` rather than equality because callers wrap the rendered text
    /// in their own context before it arrives.
    ///
    /// [`Display`]: std::fmt::Display
    pub fn rendered_is_non_pd_schur_complement(rendered: &str) -> bool {
        rendered.contains("Schur complement Cholesky failed")
            && rendered.contains("not positive definite")
    }

    /// #2515 — the phrase every RESOLVED-INDEFINITE evidence refusal carries,
    /// and the only place it is written.
    ///
    /// The two producers are the reduced-Schur and per-row conditioning under
    /// [`ArrowEvidencePolicy::UnitDeflationRefusingIndefinite`]. Their consumer
    /// is in another crate (`gam-sae` maps this to the same typed
    /// `IndefiniteObservedInformation` verdict the dense exact-`A` route
    /// returns), which is exactly the arrangement #2598 caught drifting: a
    /// reworded message in this crate silently reclassified every recoverable
    /// refusal as a fatal defect. So the wording lives beside its reader, both
    /// producers interpolate it, and [`Self::rendered_is_indefinite_evidence`]
    /// matches the same function.
    pub(crate) fn indefinite_evidence_marker() -> &'static str {
        "evidence operator carries RESOLVED NEGATIVE curvature"
    }

    /// Whether a rendered refusal is the `Self::indefinite_evidence_marker`
    /// class. `contains` rather than equality because callers wrap the rendered
    /// text in their own context before it arrives.
    pub fn rendered_is_indefinite_evidence(rendered: &str) -> bool {
        rendered.contains(Self::indefinite_evidence_marker())
    }
}

impl std::fmt::Display for ArrowSchurError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrowSchurError::PerRowFactorFailed { row, reason } => write!(
                f,
                "arrow-Schur: per-row H_tt^({row}) Cholesky failed: {reason}"
            ),
            ArrowSchurError::PerRowFactorIllConditioned {
                row,
                kappa_estimate,
            } => write!(
                f,
                "arrow-Schur: per-row H_tt^({row}) Cholesky succeeded but failed \
                 the safe-inversion guard (kappa_estimate={kappa_estimate:e}); \
                 Schur reduction would be numerically contaminated"
            ),
            ArrowSchurError::SchurFactorFailed { reason } => {
                write!(f, "arrow-Schur: Schur complement Cholesky failed: {reason}")
            }
            ArrowSchurError::PcgFailed { reason } => {
                write!(f, "arrow-Schur: Schur PCG failed: {reason}")
            }
            ArrowSchurError::PcgBudgetExhausted {
                budget,
                products_spent,
                final_relative_residual,
            } => write!(
                f,
                "arrow-Schur: Schur PCG spent its product budget ({} products, {:?}) \
                 without meeting its tolerance: {products_spent} products spent, final \
                 relative residual {final_relative_residual:e}",
                budget.products(),
                budget.basis()
            ),
            ArrowSchurError::UnboundedNegativeCurvature {
                curvature,
                direction_norm_sq,
            } => write!(
                f,
                "arrow-Schur: unbounded Schur PCG hit negative curvature pᵀSp={curvature:e} \
                 (‖p‖²={direction_norm_sq:e}); reduced Schur is indefinite (co-collapse), \
                 retry with a curvature-floor ridge"
            ),
            ArrowSchurError::AdaptiveCorrectionFailed { reason } => {
                write!(
                    f,
                    "arrow-Schur: adaptive proximal correction failed: {reason}"
                )
            }
            // The cause renders through `Debug`, so a refusal at a certified rung is
            // never read as the relocatable non-PD Schur refusal it may wrap.
            ArrowSchurError::RefusedAtCertifiedShift {
                proximal_ridge,
                cause,
            } => write!(
                f,
                "arrow-Schur: the proximal ridge {proximal_ridge:e} is certified factorable \
                 from the system's declared bounds and the solve still refused, so no larger \
                 shift can cure it: {cause:?}"
            ),
        }
    }
}

impl std::error::Error for ArrowSchurError {}

// ---------------------------------------------------------------------------
// Cholesky helpers (kept local to avoid a new public-API dependency on the
// linalg crate. The systems here are tiny per-row (d × d, d ∈ {1..16}) and
// modest at the Schur level (K × K, K ∈ {basis size}). For production SAE
// scales the Schur factor should switch to faer; this module's `cholesky_lower`
// is the obvious replacement site.)
// ---------------------------------------------------------------------------

pub(crate) fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(format!("cholesky_lower: non-square {}×{}", n, a.ncols()));
    }
    if let Some((idx, _)) = a.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "cholesky_lower: non-finite entry at linear index {idx}"
        ));
    }

    // CPU factorization seam (#1017): device routing happens explicitly in the
    // arrow-Schur solve before reaching this reference/fallback primitive. At
    // the SAE border width the reduced Schur is a
    // dense `k×k` (k≈2k–4k) whose scalar triple-loop factorization is O(k³/3)
    // and neither blocked nor SIMD-vectorized — the dominant per-Newton-step
    // cost on a CPU-only host. faer's blocked LLT computes the SAME `A = L Lᵀ`
    // (to O(κ·ε), the slack the reduced solve/log-det already tolerate) an order
    // of magnitude faster. Restrict it to `k ≥ FAER_CHOLESKY_MIN` so the many
    // tiny per-row `d×d` blocks (d≤~8, factorization.rs) and the small dense
    // test fixtures keep the exact scalar loop — bit-for-bit their historical
    // factor — where faer's setup overhead would not pay off anyway. If faer
    // declines (a non-PD blocked pivot) fall through to the scalar loop so the
    // PD/non-PD verdict and its typed error stay exactly the historical ones
    // (`factor_dense_reduced_schur`'s spectral-floor fallback keys only on Ok vs
    // Err, so the boundary behavior is unchanged).
    const FAER_CHOLESKY_MIN: usize = 128;
    if n >= FAER_CHOLESKY_MIN {
        let view = gam_linalg::faer_ndarray::FaerArrayView::new(a);
        if let Ok(llt) = gam_linalg::faer_ndarray::FaerLlt::new(view.as_ref(), faer::Side::Lower) {
            let l_faer = llt.lower();
            let mut l = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..=i {
                    l[[i, j]] = l_faer[(i, j)];
                }
            }
            return Ok(l);
        }
    }

    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for kk in 0..j {
                sum -= l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(format!(
                        "non-PD pivot {sum} at index {i} (matrix is not positive definite)"
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}
