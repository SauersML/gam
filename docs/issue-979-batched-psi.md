# Issue 979: batch mixed spatial information derivatives

Commit `edd2c22ed` replaces the rigid binary workspace's separate coefficient
direction sweeps with one row pass. It reuses cached third and fourth primary
derivatives, reads spatial design rows in blocks, and accumulates each distinct
symmetric coefficient triple once in flat buffers. Equal permutations are filled
after summation. The requested output has the same cubic storage as before;
there are no per-worker copies of that tensor.

Paired measurements on MSI acn112, 1,500 rows and 24 coefficients, using four
Rayon threads, one BLAS thread, optimization level 1, and warm row caches:

| Implementation | Batch | Existing per-axis calculation |
| --- | ---: | ---: |
| Initial multidimensional accumulation | 567.77 ms | 114.97 ms |
| Flat buffers | 57.58 ms | 106.81 ms |
| Ordered triples and blocked spatial rows (committed) | 21.35 ms | 83.28 ms |

The final paired speedup is **3.9x**. These are kernel measurements, not evidence
that either complete fit now meets the issue's performance requirements.

The new tests cover both spatial blocks with unequal coefficient widths,
nonuniform observation weights, full and subsampled outer row measures, and
normal and empirical latent distributions. Agreement with the existing exact
directional calculation is below `2e-12` relative error; coefficient finite
differences agree within `2e-7`. The tightened outer-Hessian regression also
passes in the current working implementation, with maximum relative error
`2.092e-9` across pure penalty, pure spatial, and combined directions. That
regression includes the previously documented pending analytic corrections;
the batching commit alone does not publish those corrections.

All three final checks pass in 0.92 seconds. The warm model test build took
2m56s. Logs are under `/projects/standard/hsiehph/sauer354/y1-logs/`:

- `codex979-batched-psi-tests.log`
- `codex979-batched-psi-flat-tests.log`
- `codex979-batched-psi-final-tests.log`

The issue remains open pending completed larger binary and survival fits,
remaining derivative publication, and full performance and interface validation.

## Full-fit follow-up

The rebuilt example (warm build 2m29s) ran `repro979_margslope 1500 12 1`
under a 30-second diagnostic cap. Spatial gradient assembly now takes
`0.062s` in representative late evaluations, versus `0.395–0.409s` in the
previous trace. Complete value-and-gradient evaluations take `0.229–0.261s`,
versus `0.663–0.692s` previously. These are trace comparisons, not paired
complete-fit timings.

The fit still exits at the diagnostic cap (`124`). It reaches outer evaluation
39, with gradient norm `2.222e-2`; the subsequent dense outer-Hessian evaluation
takes `3.519s` (the containing criterion call takes `3.917s`). This identifies
the next remaining cost rather than establishing convergence. Warm inner solves
at that point converge in two cycles, with residual `6.538e-11`.

Log: `codex979-batched-binary1500-c12.log` in the same MSI log directory.
The diagnostic process terminated and no owned build or fit remains running.

## Next Hessian change: validation pending

The third-information producer in `bms/information_third.rs` still updated all
equal coefficient permutations through multidimensional indexing on every row.
The working implementation now accumulates its ordered triples in flat buffers
and expands the equal entries after summation, using the same algebraic symmetry
as the verified mixed-derivative batch.

New checks differentiate the mixed information drift along a coefficient
direction and along the other spatial design block, for both normal and
empirical latent distributions. The wide timing fixture also measures this
third-information calculation. These edits are **not yet compiled or tested**.

MSI validation is currently unavailable: direct compute access failed for
acn112, acn116, and the newly listed Sioux node acl42. The login node is
reachable, but `sinfo` reports invalid/unknown compute states and a read-only
`sbatch --test-only` request for four CPUs and 16 GiB on Sioux reports
`Requested node configuration is not available`. No job was submitted, and no
build or numerical test was run locally or on the login node.
