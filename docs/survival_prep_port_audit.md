# Survival-Flex Prep-Work NVRTC Port — CPU-Side Audit

Scope: port two per-row CPU hot-spots in `src/families/survival_marginal_slope.rs`
to NVRTC so that biobank-scale survival-flex fits stop being bottlenecked by
prep work. The two functions are:

1. `SurvivalMarginalSlopeFamily::denested_partition_cells`
   (`src/families/survival_marginal_slope.rs:5718`)
2. `SurvivalMarginalSlopeFamily::denested_cell_primary_fixed_partials`
   (`src/families/survival_marginal_slope.rs:6235`)

Both are family-private methods called per row, per timepoint (entry + exit)
inside `compute_row_flex_primary_gradient_hessian_from_parts`
(`src/families/survival_marginal_slope.rs:7263`).  They are consumed by the
cached-partition builder `build_cached_partition_with_moment_order`
(`src/families/survival_marginal_slope.rs:6541`), which loops every raw
partition cell and produces a `CachedCellEntry { partition_cell, neg_cell,
state, fixed }` (`src/families/survival_marginal_slope.rs:953`).

The downstream Layer-A/B/C-α/C-β device entries already consume row-primary
G/H assemblies (closed by survival-flex-v3 Steps 4 + 5).  The remaining CPU
cost is the prep that produces the per-cell `coeff_*` tables and the cell
partition itself.

---

## 1. `denested_partition_cells`

### Inputs

| Name | Type | Source |
|------|------|--------|
| `a` | `f64` | row affine intercept (`a0`/`a1` from `solve_row_survival_intercept_with_slot`, `src/families/survival_marginal_slope.rs:7284` / `:7291`) |
| `b` | `f64` | row slope `g` (the per-row log-slope) |
| `beta_h` | `Option<&Array1<f64>>` | full score-warp β |
| `beta_w` | `Option<&Array1<f64>>` | full link-deviation β |

Score/link runtime objects (`score_warp`, `link_dev`) live on `self`, and
`probit_frailty_scale()` is queried at end of build.

### Branching shape

```
if self.score_dim() == 1:
    delegate → marginal_slope_shared::build_denested_partition_cells
              (single-coord fast path, src/families/marginal_slope_shared.rs:102)
else:
    score_breaks = score_warp.breakpoints()
    link_breaks  = link_dev.breakpoints()
    cells = cubic_cell_kernel::build_denested_partition_cells_with_tails(
                a, b, &score_breaks, &link_breaks,
                score_span_at = |z| self.score_warp_local_cubic_at(beta_h, z)   // sums score_dim coords
                link_span_at  = |u| link_dev.local_cubic_at(beta_w, u)
            )
    scale = self.probit_frailty_scale()
    if scale != 1.0: in-place rescale of (c0, c1, c2, c3) on every cell.cell
```

Both branches return `Vec<DenestedPartitionCell>`
(`src/families/cubic_cell_kernel.rs:1071`) — one `DenestedCubicCell` plus the
`LocalSpanCubic` score / link spans used to derive it.

### The branch-heavy block-state logic that makes the port nontrivial

`build_denested_partition_cells_with_tails`
(`src/families/cubic_cell_kernel.rs:2637`):

1. **Split-point assembly** — merge `score_breaks` with `{ (τ − a) / b : τ ∈ link_breaks }`
   when `|b| > 1e-12`, then `dedup_sorted_breakpoints`.  Empty split-point
   list → single-cell fast path covering `(-∞, +∞)`.

2. **Left-tail cell** `(-∞, leftmost_split]`:
   - probe at `interval_probe_point(NEG_INF, leftmost)`,
   - require `|c2| ≤ NORMALIZED_CELL_BRANCH_TOL && |c3| ≤ NORMALIZED_CELL_BRANCH_TOL`
     otherwise hard error.

3. **Interior cells** — windowed pairs, skip degenerate width `≤ 1e-12`,
   probe at midpoint, evaluate `denested_cell_coefficients(score_span, link_span, a, b)`.

4. **Right-tail cell** `[rightmost_split, +∞)` — same affine-only requirement.

The score-warp closure for `score_dim > 1` (`score_warp_local_cubic_at`,
`src/families/survival_marginal_slope.rs:4124`) itself iterates `score_dim`
coordinates: extracts the per-coord β via `score_warp_component_beta`,
evaluates `runtime.local_cubic_at`, and **sums** the spans coordinate-by-
coordinate.  The first coordinate's `(left, right)` is adopted as the span
support.

### Branch-heavy aspects, ranked by NVRTC-port pain

| Source of branching | Notes for device port |
|---------------------|----------------------|
| `score_dim() == 1` vs multi-coord | Specialize two NVRTC kernels OR factor span evaluation into a coord-loop with `MAX_SCORE_DIM` constant |
| Split-point merge + dedup | Sort + unique in shared memory; bound by `MAX_SCORE_BREAKS + MAX_LINK_BREAKS` |
| `|b| ≤ 1e-12` skip of link breaks | Cheap branch, no warp divergence concerns |
| Tail-affineness validation | Device kernel must emit a per-row status byte ("invalid_cell_shape") consumed by host fallback (mirrors `RidgeBumpRequired` pattern) |
| Probe-point selection at tails | `interval_probe_point(NEG_INF, x)` / `(x, INF)` needs an explicit infinite-tail branch — small fixed-shape divergence |
| `scale != 1.0` in-place rescale | Trivial post-pass on flat coeffs |
| Cells per row variable length | Flat-packed `DeviceCellTable` with per-row `cell_offsets[n_rows+1]`; `max_cells_per_row` upper-bounded by `score_breaks.len() + link_breaks.len() + 2` |

### Outputs

```rust
Vec<DenestedPartitionCell>
  // DenestedPartitionCell {
  //     cell: DenestedCubicCell { left, right, c0, c1, c2, c3 },
  //     score_span: LocalSpanCubic { left, right, c0, c1, c2, c3 },
  //     link_span:  LocalSpanCubic { left, right, c0, c1, c2, c3 },
  // }
```

Device layout proposal: `f64[18]` per cell (6 + 6 + 6 packed in
`DenestedPartitionCell` order), plus one `i32` per row for the cell count and
one `u8` for the per-row status.

---

## 2. `denested_cell_primary_fixed_partials`

### Inputs

| Name | Type | Source |
|------|------|--------|
| `primary` | `&FlexPrimarySlices` | row-primary parameter layout (`src/families/survival_marginal_slope.rs:763`) |
| `a`, `b` | `f64` | timepoint intercept and slope |
| `score_span`, `link_span` | `LocalSpanCubic` | output of partition step |
| `z_basis` | `f64` | per-cell probe `z_mid` (== `interval_probe_point(cell.left, cell.right)`) |
| `u_basis` | `f64` | `a + b * z_mid` |

`self` carries `score_warp`, `link_dev` runtimes plus `probit_frailty_scale()`.

### What it builds

For a single cell it constructs ten `Vec<[f64; 4]>` of length `r =
primary.total`, where `r = 4 + h.len() + w.len()` and the slots are laid out
by `FlexPrimarySlices`:

```
primary.q0  = 0      // q at entry
primary.q1  = 1      // q at exit
primary.qd1 = 2      // q-derivative at exit
primary.g   = 3      // per-row log-slope
primary.h   = Some(4 .. 4 + score_warp.basis_dim() * score_dim)   // score-warp β slot
primary.w   = Some(end_of_h .. end_of_h + link_dev.basis_dim())   // link-deviation β slot
```

Body breakdown (`src/families/survival_marginal_slope.rs:6235`):

1. **Cell-affine derivatives w.r.t. (a, b)** —
   `denested_cell_coefficient_partials` (1st), `denested_cell_second_partials`
   (2nd), `denested_cell_third_partials` (3rd; link-only).  Each is a `[f64; 4]`
   (the cubic coefficient vector); each gets `scale_coeff4(_, scale)`.

2. **Plant g-slot entries** — write the link-side partials into the
   `primary.g` row of each table (`coeff_u[g]`, `coeff_au[g]`, ...,
   `coeff_bbu[g]`); the third-order `coeff_aaau/aabu/abbu/bbbu[g]` are zero
   (the link cubic is independent of `b`-via-`g` at this order; only the
   `(a,b,b,…)` chain is loaded).

3. **Score-warp h-slot block** — when `primary.h` exists and `score_warp`
   non-null, for each `local_idx ∈ 0 .. h.len()`:
   - `coeff_u[h.start + local_idx] = scale · score_basis_cell_coefficients(basis_cubic_at(basis_idx, z_basis), b)`
   - `coeff_bu[h.start + local_idx] = scale · score_basis_cell_coefficients(basis_cubic_at(basis_idx, z_basis), 1.0)`
   - All other tables: zero.
   - The `(coord, basis_idx) = score_warp_coord_basis_index(local_idx)` decomposition
     is identity when `score_dim == 1`; for multi-coord it picks the basis
     within the active coordinate (the cross-coord contribution is in the
     **partition** stage, not here).

4. **Link-deviation w-slot block** — when `primary.w` exists and `link_dev`
   non-null, for each `local_idx ∈ 0 .. w.len()`:
   - `basis_span = link_dev.basis_cubic_at(local_idx, u_basis)`
   - `coeff_u   = scale · link_basis_cell_coefficients(basis_span, a, b)`
   - `coeff_{au,bu}        = link_basis_cell_coefficient_partials(basis_span, a, b)`
   - `coeff_{aau,abu,bbu}  = link_basis_cell_second_partials(basis_span, a, b)`
   - `coeff_{aaau,aabu,abbu,bbbu} = link_basis_cell_third_partials(basis_span)`

5. **Return** `DenestedCellPrimaryFixedPartials`
   (`src/families/survival_marginal_slope.rs:919`) — a struct of 13 fields:
   the three `dc_da/daa/daaa` (the unit `[f64; 4]` partials over the cell
   coefficients themselves) plus the ten `Vec<[f64; 4]>` per-row tables.

### Outputs flattened for device

Per (row, cell) the flat layout consumable by Layer A/B/C entries is:

```
dc_da[4], dc_daa[4], dc_daaa[4]                       // 12 doubles
coeff_u[r][4]                                          // 4r doubles
coeff_au[r][4]   coeff_bu[r][4]                        // 8r
coeff_aau[r][4]  coeff_abu[r][4]  coeff_bbu[r][4]      // 12r
coeff_aaau[r][4] coeff_aabu[r][4] coeff_abbu[r][4] coeff_bbbu[r][4]  // 16r
                                                       // total = 12 + 40·r doubles
```

with `r = primary.total = 4 + h.len() + w.len()`.

### Branch-heavy aspects, ranked

| Branch | NVRTC handling |
|--------|---------------|
| g-slot scalar writes | Pure straight-line per cell |
| h-slot loop (score_warp present?) | Mask via `h_len = 0` when absent; otherwise one thread per (cell, h_local_idx) |
| w-slot loop (link_dev present?) | Same mask pattern with `w_len` |
| `score_warp_coord_basis_index` | When `score_dim == 1`, identity; multi-coord requires a small `coord = idx / basis_dim` divmod — device-cheap |
| `scale != 1.0` post-scale | Fold `scale` into a single multiplier per write site, no branch needed |
| Per-cell basis-cubic queries | Need device-side `score_warp.basis_cubic_at` and `link_dev.basis_cubic_at`. These already exist in CPU runtimes; the port needs a device-resident knot-vector / coefficient table for both runtimes. Existing `src/gpu/cubic_cell` substrate covers `LocalSpanCubic` evaluation — extend with `basis_cubic_at` indexed lookup. |

---

## 3. `FlexPrimarySlices` layout dependency

The port must consume the same `FlexPrimarySlices` layout that the CPU path
already uses, both because the downstream Layer A/B/C entries are indexed by
`primary.q0`, `primary.q1`, `primary.qd1`, `primary.g`, `primary.h`,
`primary.w` and because `compute_row_flex_primary_gradient_hessian_from_parts`
addresses `grad[u]` / `hess[[u,v]]` with those slot indices
(`src/families/survival_marginal_slope.rs:7347`, `:7351`, `:7368`, `:7375`).

What the GPU kernel needs to receive (constant for the whole call, not
per-row):

```
q0, q1, qd1, g           // four u32 indices (compile-time after launch)
h_start, h_len           // u32; h_len = score_warp.basis_dim() * score_dim
w_start, w_len           // u32; w_len = link_dev.basis_dim()
r = total                // u32; r = 4 + h_len + w_len
score_dim                // u32; ≥ 1
score_warp_basis_dim     // u32 — used by score_warp_coord_basis_index
```

These pack as a single `__constant__ FlexPrimaryLayout` struct (≤ 64 bytes).

Per-row scalars (entry + exit each):

```
a0, a1   // f64
g        // f64 (per-row log-slope)
z_basis, u_basis // recomputed device-side from cell midpoint + (a,b)
```

Per-row vectors (shared across cells):

```
beta_h[h_len]    // optional, mask if absent
beta_w[w_len]    // optional, mask if absent
```

The downstream Layer A/B/C entries already expect a row-primary G/H assembly
indexed in this exact layout, so the device-side output of the prep kernel can
flow straight in.

---

## 4. Plan for Steps 2 – 7

The port splits cleanly:

- **Step 2 — `denested_partition_cells_kernel`**: one thread per row.  Inputs:
  per-row `(a, b)`, optional `beta_h` / `beta_w`, plus constant
  `(score_breaks, link_breaks, score_warp_runtime, link_dev_runtime)`
  packaged as device tables.  Output: flat `DeviceCellTable` with per-row
  `cell_offsets[n_rows + 1]`, plus a per-row `status[u8]` (0 = ok, 1 =
  invalid tail shape, 2 = singular-b skip, …) for host fallback.

- **Step 3 — `denested_cell_primary_fixed_partials_kernel`**: one thread per
  `(row, cell)` (or per `(row, cell, slot)` if profiling justifies more
  parallelism).  Consumes the cell table from Step 2 plus the
  `FlexPrimaryLayout` constant.  Output: the flat `12 + 40·r` doubles per
  cell, packed contiguously into a `DeviceFixedPartialsTable` aligned with
  the cell table (same row/cell offsets).

- **Step 4 — CPU oracles + parity**: byte-identical Rust ports of both
  kernels (so we can drive a 4-row × 3-cell fixture without GPU), parity to
  `1e-13 abs` on a fixture with `score_dim ∈ {1, 2}` × `link_dev present /
  absent` × `scale ∈ {1.0, 0.7}`.

- **Step 5 — V100 parity**: cross-check device output against CPU oracle to
  `1e-12 rel` on the same fixtures.  Skip locally; runs under
  `#[cfg(all(test, target_os = "linux"))]`.

- **Step 6 — Wiring**:
  `compute_row_flex_primary_gradient_hessian_from_parts` already calls
  `compute_survival_timepoint_exact` (`src/families/survival_marginal_slope.rs:7298`),
  which routes through `build_cached_partition_with_moment_order`.  Insert a
  "try GPU prep first" branch at the top of `build_cached_partition_with_moment_order`
  that:
    1. Calls `try_device_partition_cells` (host wrapper around Step 2 kernel).
    2. On success, calls `try_device_cell_primary_fixed_partials` (Step 3).
    3. Produces a `CachedPartitionCells` whose `state` is still built by
       `evaluate_cell_moments` host-side (that path is the survival-flex-v3
       Layer A device entry — already on GPU when the dispatcher accepts it).
    4. On any per-row `status != 0`, fall back to the existing CPU per-row
       loop for those rows only (sub-batch fallback, not whole-call).

  End-to-end family-level parity test: `survival_marginal_slope_flex_gpu_prep_parity_vs_cpu`
  at `(n = 4096, p_primary ≈ 24)` to `1e-10 rel`.

- **Step 7 — Hill-climb**: `survival_flex_v100_end_to_end_hill_climb_10x_dense_h_5x_hvp`
  at biobank shape `(n = 320_000, p_primary ≈ 100)`.  Profile bottleneck if
  miss; the candidates already on the table are (a) score-warp basis-cubic
  device cache size, (b) per-row `cell_count` variance causing warp
  divergence — switchable to one-warp-per-row.

---

## Open questions to resolve before Step 2

- Does the score-warp runtime already expose a device-side `basis_cubic_at`?
  `src/gpu/cubic_cell/{device,kernel_src}.rs` evaluates a `LocalSpanCubic`
  device-side but does **not** appear to expose a knot-vector-indexed
  `basis_cubic_at`.  Step 2 will need to ship the runtime's knot vector +
  per-basis coefficient table as constant device memory and add a small
  `basis_cubic_at_device` helper.  No new feature flag; auto-derive layout
  from `score_warp.basis_dim()`.

- Score-warp coordinate beta extraction (`score_warp_component_beta`,
  multi-coord case): trivial slice on host, trivial pointer arithmetic on
  device.  No new API surface required.
