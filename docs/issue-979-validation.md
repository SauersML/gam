# Issue 979 validation

The issue remains open. Binary derivative correctness and survival convergence
are separate acceptance conditions; a passing derivative check does not establish
that either full fit is fast.

## September 8 investigation

- The September 7 cached model test binary passes the previously failing
  `profiled_theta_hvp_outer_hessian_matches_fd_of_gradient_psi_and_mixed`
  reproduction in 0.51 seconds with four Rayon threads on acn112. This is evidence
  for the unfinished analytic implementation, not current-source validation.
- Nine related tests in that cached binary pass in 4.58 seconds. They include
  Gaussian-measure and empirical-measure BMS fifth derivative checks, the survival
  all-axis derivative contract, and the survival contracted Jeffreys trace check.
  The latter agrees with its pairwise definition to relative error `5.718e-14`
  at `n=800`, `p=41` (one pass versus 861 pair passes).
- The cached survival example at `n=160`, six Duchon centers still exceeds a
  20-second diagnostic cap. Its inner solver reaches cycle 130, with projected
  residual `9.125e-2` against tolerance `2.382e-9`; successive accepted steps
  decrease that residual by only about two percent. This reproduces a convergence
  problem independently of the binary outer-Hessian defect. The process was
  stopped at the diagnostic cap.
- Current-source model validation exposed a dependency cycle:
  `gam-models` (tests) -> `gam-test-support` -> `gam-models` (library).
  The only model dependency in the support crate is one diagnostics constructor.
  Passing it design geometry, smoothing parameters, and effective degrees of
  freedom removes the cycle and the duplicate model compilation.
- The new mixed Jeffreys derivative now assembles its spectral linear map one
  output row at a time and contracts all coefficient axes using dense matrix
  products. Scratch is `m^3` entries, within the existing `O(p m^2)` axis storage
  since the reduced dimension `m <= p`; no `m^4` tensor is allocated.
- First and second coefficient derivatives now share a lazy, immutable Jeffreys
  base across callback invocations. This avoids repeated first-information
  derivative sweeps on every operator application.

## Current-source results

The binary outer-Hessian reproduction now passes in every tested component.
This includes pure penalty, pure spatial, and combined directions, using both
the operator and its dense materialization. The maximum relative error is
`2.092e-9` (pure-spatial diagonal: analytic `3.777175822301`, finite difference
`3.777175812305`). The current-source numerical test takes 2.26 seconds with
four Rayon threads. The acceptance tolerance remains `2e-6`.

The causal sequence was:

1. Use the stationarity inverse for the second mode response, as for the first.
2. Differentiate the difference between that Jacobian and the logdet matrix in
   the response right-hand side. This reduced all penalty and cross components
   to approximately `1e-10` error.
3. Include the missing explicit spatial–spatial derivative of the inner
   Jeffreys score, using analytic third derivatives of the scalar spectral term.
   This removed the remaining `1.194e-4` spatial-diagonal error.

The generic second-response solve and correction contract are pushed to main
in `5e2cc8a74` (merged with concurrent main updates in `d41828c40`). The
independent test with different stationarity and logdet matrices passes for
both Hessian implementations. The family and spectral pieces are still being
checked and prepared for publication. Full-fit performance is not yet verified.

- The tightened binary outer-Hessian check exposed a remaining relative error
  of `4.178e-5` in the first pure-psi component: analytic
  `-0.15917195570456882`, finite difference `-0.15922038488207224`.
  The former `2e-3` tolerance hid this error; the acceptance tolerance is `2e-6`.
- The mixed spectral derivative check initially failed at the repeated spectrum
  `[0.4, 2, 2]` with error `1.068e-3`. Replacing the first derivative's
  eigenvector-rotation formula with fixed-frame Frechet divided differences,
  and using rational divided differences for the base weights, fixes this
  check. Gate transition, repeated eigenvalues, moving relative floor, and
  signed continuation cases now pass together in 0.16 seconds.
- Current-source fifth-order likelihood checks pass for normal and empirical
  BMS, multinomial, and cause-specific survival. The normal log-CDF fifth
  derivative also passes finite differences and high-precision references.
  The wrapper's bilinear Jeffreys correction check passes.
- The spectral correction leaves the binary outer error unchanged. Inspection
  found that the second mode response still solves with the logdet operator,
  although the first response correctly uses the stationarity operator. Both
  dense and matrix-free routes are being corrected and checked independently.

Current-source tests use MSI's existing `y2-target` cache, eight build workers,
four Rayon threads, and one BLAS thread. The model crate uses 16 codegen units
and optimization level 0 during iteration; these runs are correctness evidence,
not release-performance measurements. Builds and logs are on shared project or
scratch storage. Nothing is built or numerically executed on the local Mac.

Logs are in `$MSI_HOME/y1-logs/`:

- `codex979-cached-contracts.log`
- `codex979-survival-baseline6.log`
- `codex979-contracts-acyclic.log`
- `codex979-correctness-seed.log`
- `codex979-fixed-frame.log` (current build 1m59s, numerical checks under 2s)
- `codex979-completion-response2.log`
- `codex979-scalar-third.log` (binary regression passes; new spectral check
  exposes an overbroad rejection of repeated interior eigenvalues)

Pending: current-source derivative checks at the tightened `2e-6` outer gate,
full binary and survival fits, release-performance checks, and publication.
