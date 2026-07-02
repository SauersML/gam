# Batched independent K=1 border solves — the new GPU target (WS-G)

Status: design + skeleton. The dispatch entry, the CPU reference/oracle path, and
the parity/decline tests land now in `batched_k1.rs`; the device batched kernel is
gated behind the Linux/CUDA build and currently declines to `Unavailable` (→ CPU
reference) so the seam is correct and buildable on every host today.

## 1. Why this replaces the old GPU job

The pre-SAC GPU ambition was to run the **giant joint K-atom arrow-Schur system**
on device: one bordered Hessian with a `K × K` shared β-block, `K` = the whole T2
dictionary (thousands to tens of thousands of atoms). That system is dead — not
because of the GPU, but because the *training path* that produced it is dead. The
joint cold-start fit of K atoms co-collapses on real activations (SAC_PLAN Part 1);
SAC retires it from training and demotes the joint solver to a single
evaluate-don't-optimize terminal pass (Phase 3). See §7 for the explicit obituary.

What SAC's backfitting phase (Phase 2) actually needs is the opposite shape. After
the forward births produce K single-atom charts, backfitting re-solves each atom
against its leave-one-out residual. Atoms whose supports do not overlap are
mutually independent — refitting one cannot change another's block of the joint
penalized objective. So we **color atoms by support overlap and refit each color
class concurrently** (SAC_PLAN Phase 2, WS-G line). A color class is a *batch of B
independent, small K=1 arrow/border systems*: embarrassingly parallel, each tiny,
no cross-atom coupling. That is the workload the B200s are good at, and it is
exactly the batched-per-block structure the existing arrow-Schur device kernels
(`cuda::solve`, `cuda::solve_fused`) were built around — only the batch axis moves
from "rows within one system" to "independent atoms".

## 2. The per-atom system (what one batch element is)

One backfitting refit of atom `a` is a K=1 `ArrowSchurSystem` (`arrow_schur/system.rs`):

- `rows`: one `ArrowRowBlock` per **active** row in atom `a`'s support `S_a`
  (`n_a = |S_a|`). Each carries `H_tt^(i)` (`d × d`, `d = d_atom`, typically 2 for a
  circle), `H_tβ^(i)` (`d × k_a`), and `g_t^(i)` (`d`).
- `hbb`: the atom's own dense border, `k_a × k_a`. For a single unit-speed chart
  under the WS-B gauge quotient `k_a` is small — the chart's decoder-frame
  coordinates plus log-amplitude, `k_a = O(d)` (a handful), **not** the global K.
- `gb`: the `k_a`-length border gradient.

The border `k_a` being tiny is the whole point. The joint system died on its `K³`
border factor (`K = 3.2e4 ⇒ K³ ≈ 3.3e13` flops for one Cholesky). A single atom's
border factor is `k_a³ ≈ 10³` flops; per-atom cost is dominated by the row work
`n_a · d²`, and the batch is `Σ_a n_a · d²` — linear in total active mass, with no
`K²`/`K³` term anywhere. The pathology that killed the joint path cannot occur here
by construction.

## 3. Memory layout

Supports are ragged (`n_a` varies atom to atom), so the batch is a jagged/CSR pack,
never a padded dense cube (padding to `max_a n_a` wastes memory and flops when
supports are imbalanced, and SPEC forbids OOM on reasonable hardware). Layout, all
device-resident, column-major to match the existing tile packers:

```
atom_offsets : [B+1]  prefix sum of n_a over the color class (row-block CSR)
row_dims     : [B]    d_a per atom (uniform d today; per-atom d kept for hetero charts)
border_dims  : [B]    k_a per atom
D_blocks     : Σ_a n_a · d·d   stacked column-major H_tt^(i) (+ ridge_t on diag)
B_blocks     : Σ_a n_a · d·k_a stacked H_tβ^(i)
g_t          : Σ_a n_a · d     stacked row gradients
hbb          : Σ_a k_a·k_a     stacked per-atom borders (+ ridge_beta on diag)
g_b          : Σ_a k_a         stacked border gradients
```

Output mirrors the input CSR: `delta_t` at `atom_offsets · d`, `delta_beta` at a
`border_offsets` prefix sum, plus one `log_det_hessian` scalar per atom. Only these
three per-atom results download; the factors stay on device (same scalars-only
readback contract as `#1017`).

## 4. Kernel / stream structure

Each atom's arrow-Schur sequence is self-contained: per-row `D_i = L_i L_iᵀ`
Cholesky → `u_i = L_i⁻¹ g_i`, `Y_i = L_i⁻¹ B_i` triangular solves → reduce
`S_a = C_a + ρ_β I − Σ_i Y_iᵀ Y_i`, `r_a = −g_b + Σ_i Y_iᵀ u_i` → factor the tiny
`S_a` → back-substitute `δβ_a`, then `δt_i`. Because a color class is
support-disjoint, **no atom touches another's memory**, so there is no cross-atom
reduction or barrier.

Two dispatch shapes, chosen by the same work heuristic the single-system paths use:

- **One block per atom (preferred at SAC scale).** A grid-strided launch assigns
  one CUDA thread-block per atom; the block runs that atom's whole sequence in
  shared memory. Because `k_a ≤ MAX_FUSED_P = 32` and `d` is tiny, each atom fits
  the fused NVRTC kernel's shared-memory budget (the fused path already admits when
  `p ≤ MAX_FUSED_P` and `Σ p³ ≥ 1e5 OR R ≥ 16`; a color class of many small atoms
  clears the `R ≥ 16` row-count arm immediately). B atoms = B resident blocks in
  one launch — a single NVRTC compile amortized over the class.
- **Stream-per-atom (fallback / very large `n_a`).** When one atom's `n_a` is large
  enough to fill a device on its own, give each atom its own stream and reuse the
  single-system `cuda::solve` per atom; the streams overlap across the class. This
  is the mode that also serves multi-GPU: split the color class round-robin across
  devices at atom granularity (the joint solve's `solve_multi_gpu` already splits at
  row-block granularity — here the split is cleaner because atoms are truly
  independent).

## 5. Break-even vs CPU

Known B200 facts to anchor the floor (from the resident-engagement work): NVRTC JIT
compiles and caches the fused kernel; a production-shaped SAE frame engaged the
device at **612 MiB resident**. The batched-K1 launch reuses that resident-frame
machinery, so the fixed cost is the same order — a one-time NVRTC compile plus the
per-class upload of `Σ_a n_a·d·(d+k_a)` f64.

The CPU reference does the class **sequentially**, Rayon-parallel over rows within
each atom (`solve_arrow_newton_step_dense_reference` per atom). It wins whenever the
class is small or the atoms are individually tiny — launch + `H2D`/`D2H` staging
then dominates. The device wins when the class has enough **total** active mass to
hide the launch: reuse `policy().reduced_schur_matvec_should_offload(Σ_a n_a, k̄, d,
1)` on the aggregate row count of the color class (CG budget = 1 because a K=1
Direct solve is one factor, not a CG loop). Concretely the device engages when a
color class has many atoms (`B` large — the common case early in backfitting, when
supports are disjoint and coloring yields big classes) or a few atoms with large
supports. Late in backfitting, when only a handful of overlapping atoms remain and
classes shrink to `B = 1–2`, the predicate declines and the CPU reference runs
bit-identically. Magic-by-default, no flag: the class shape drives the decision,
exactly like `try_device_arrow_direct`.

Decline contract, inherited from WS-12: a per-atom capability mismatch (a matrix-free
atom, or a device-transient) declines that **atom** to the CPU reference; it never
fails the class and never surfaces a fatal `RemlConvergenceError`. A genuine
numerical PD failure on an atom (`RidgeBumpRequired` / `SchurFactorFailed`) is
returned per-atom so A2's sweep can bump that atom's ridge — the same
recoverable-vs-fatal split the single-system seams use.

## 6. The API seam A2's backfitting sweep calls

A2 owns `stagewise.rs` (the `fit_stagewise` driver: residual/Σ/nursery state, birth
race, coloring, terminal assembly). WS-G owns the batched solver. The seam is one
function; A2 assembles per-atom systems from the leave-one-out residuals (he owns
that state) and hands a **color class** — a slice of mutually support-disjoint K=1
systems — to:

```rust
/// Refit a color class of mutually support-disjoint K=1 atoms concurrently.
/// `systems[a]` is atom a's leave-one-out arrow/border system. Returns one
/// result per atom, positionally aligned with `systems`. A per-atom capability
/// decline is served by the CPU reference transparently (that element is still
/// `Ok`); only a genuine numerical PD failure yields a per-atom `Err` the caller
/// escalates by bumping that atom's ridge.
pub fn solve_batched_k1_border(
    systems: &[ArrowSchurSystem],
    ridge_t: f64,
    ridge_beta: f64,
) -> Vec<Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure>>;
```

Rationale for the signature:
- **Input is `&[ArrowSchurSystem]`, not a bespoke packed struct.** A2 already builds
  `ArrowSchurSystem`s for the single-atom fits; reusing the type means the CPU
  reference is literally the proven `solve_arrow_newton_step_dense_reference` in a
  loop, and the device packer's job is purely to gather these into the CSR layout of
  §3. The internal `BatchedK1System` packed form (device-side) is an implementation
  detail behind this entry, not part of the seam.
- **Output is positional per-atom `Result`.** Coloring guarantees independence, so
  results compose without interaction; A2 writes each accepted `(δt, δβ)` back into
  its atom's chart/gate exactly as the Python prototype's Phase 2 does
  (`examples/sac_prototype.py` lines 246–279), and re-races only the atoms that
  returned `Err`.
- **Ridges are class-uniform.** Backfitting runs at fixed smoothing within a sweep
  (SAC_PLAN: "ρ moves via EFS *between* sweeps"), so one `(ridge_t, ridge_beta)` per
  class is correct; per-atom escalation is handled by re-calling with that atom's
  bumped ridge.

A2 does **not** need to know whether a class ran on device or CPU — the numbers are
identical (the device path is validated to bit-parity against the reference in the
`#1017`/`#1551` sense). WS-G does not edit `stagewise.rs`; this doc *is* the seam
contract.

## 7. What GPU is explicitly NOT for anymore

The giant joint multi-atom arrow-Schur system — one bordered Hessian with a dense
`K × K` β-block spanning the entire T2 dictionary — is no longer a GPU target,
because it is no longer a target at all. It died with the joint training path: SAC
never assembles it during fitting (atoms are born and refit one at a time), and the
only surviving joint solve is Phase 3's single terminal evidence pass, run in
`#850` evaluate-don't-optimize mode at an already-converged point. That pass forms
`½log|H|`, the cross-atom covariance, and the identifiability report **once**; it is
latency-insensitive, memory-bound on a `K × K` (or SLQ-reduced) object, and runs on
the CPU (or the existing streaming/SLQ evidence lane for large K). Chasing it onto
the GPU would resurrect the exact `K³` border factor whose cost and co-collapse
motivated SAC. The GPU's arrow-Schur job is now *only* the batch of small,
independent, per-atom border solves described above; the monolith is retired.
