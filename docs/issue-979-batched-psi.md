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
