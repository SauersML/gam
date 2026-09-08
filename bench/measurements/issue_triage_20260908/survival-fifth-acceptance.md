Commit `f4b2565af` supplies exact fifth information derivatives for rigid static
and follow-up-varying survival slopes. MSI validation passed 34 focused tests:

- [2 row tensor checks](survival-fifth-tensor-tests.log): every entry of the
  fifth derivative tensors for the static four-primary and dynamic six-primary frames
  agrees with finite differences of the canonical fourth-order row evaluator.
  Both event and censoring rows are covered, including nonzero slope rates.
- [9 follow-up domain/value checks](survival-follow-up-fifth-current.log).
- [21 parameter derivative checks](survival-psi-fifth-current.log).
- [2 normal-CDF fifth derivative checks](survival-fifth-math-tests.log), including
  independent high-precision tail references and a representable subnormal.

The row value comparison consumed only the scalar likelihood from both
evaluators. For 16,384 rows, the Newton evaluator took 58.043392 ms and the
scalar evaluator took 13.979897 ms (4.152 times faster). This measures the row
kernels in an unoptimized final test crate; it is not a full-fit speedup claim.

The canonical `gam-models` unit binary was compiled on MSI from the shared
source with final-crate optimization level zero, 256 codegen units, and four
CPUs. Its exact production dependency graph came from the completed native
`gam-models-93f85675682a118c` Cargo fingerprint. Dependencies were preserved with
hardlinks after concurrent shared-cache deletion interrupted an earlier link.
The successful binary is `.buildd/gam-models-fifth-current`; its source hashes,
dependency SHA256 values, and compiler arguments are recorded remotely in
`.buildd/survival-fifth-snapshot.provenance.json`. The separate math unit's
provenance is `.buildd/gam-math-fifth.provenance.json`.

These checks validate the derivative path and value/domain behavior. They do
not certify full survival-fit convergence or close #2767, #2695, or #2714.
The public fit gates remain under investigation. No local build or test ran.
