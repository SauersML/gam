# Issue #1017 — device-resident SAE joint fit (the 1e4–1e6× hardware gap)

Autonomous WIP worklog. This is the project's multi-week GPU/systems tracking issue.
It is NOT a closeable bug; it is a phased architectural program. Prior fleet work
(see issue comments) has landed Phases 0–1 (process-parallel candidates, dispatch
re-keying), the CPU-resident factored reduced-Schur operator, the production
auto-selection seam, and measured A100 benchmarks (PIRLS XᵀWX ~60×; full Arrow-Schur
solve ~0.94× — the dense K×K border is the bottleneck).

## Constraint on THIS box
No GPU (no nvidia-smi, no libcuda). GPU wall-clock is off-table. Focus areas that are
verifiable on a CPU-only host:
- Correctness/parity hardening of the device-residency CPU-reference paths.
- Dispatch/offload predicate logic (`reduced_schur_matvec_should_offload`, etc.).
- The verification battery (bit-parity / criterion-ranking invariants).
- Killing the dense K×K border materialisation — the measured bottleneck — so the
  GPU-favorable n-row work dominates (the scoped next rung from the A100 benchmarks).

## Plan
1. Map current state of arrow_schur reduced-solve + sae_resident + gpu policy.
2. Find the concrete, CPU-verifiable next rung (likely: block-sparse / factored
   border so the dense p×p Cholesky shrinks) and harden it with bit-parity tests.
3. Strengthen the verification battery for criterion-ranking invariance.
