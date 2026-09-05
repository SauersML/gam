# Spatial interpolation setup (#2827)

The 518 design realizations reported in [#2827](https://github.com/SauersML/gam/issues/2827)
are 513 fixed Chebyshev interpolation nodes, three off-node checks, one seed
re-realization, and one exact-polish re-realization. The expensive phase precedes
the optimization.

`setup_vs_optimizer.csv` extracts the two phases from the retained `psilog.log`
trace of the Gaussian 50,000-row, six-dimensional, 100-center Duchon fixture.
Source trace SHA-256: `fb1a9b408cccc0de045968186b183d6dafe5c777f6e2af549a6802005f272edc`.
The first 513 logged coordinates fit the source's cosine-node formula with a
maximum residual of 5.33e-7 in the six-decimal stage output. The fitted window is
approximately [-0.511102708, 5.686759129].

The actual optimizer starts at psi=0 and its first trial reaches psi=-0.5111.
The winning BFGS solve takes 15 iterations and 3.763 seconds. In this trace,
interpolation setup occupies approximately 277 seconds, from the first design
realization at 11 seconds to attachment at 4 minutes 48 seconds. These are
measurements from the original trace, not timings of the adaptive repair.

![Setup nodes and actual optimizer callbacks](setup_vs_optimizer.png)

Reproduce the extraction and plot with NumPy and Matplotlib:

```sh
python plot.py INPUT.log setup_vs_optimizer.png setup_vs_optimizer.csv
```

The adaptive repair uses nested Lobatto nodes and retains only exact Gram/RHS
samples. It requires coefficient tails to reach their floating-point
accumulation floor and checks both statistics against exact off-node rebuilds.
The 16 tensor tests pass, including analytic derivative, ridge-penalized solve,
profile curvature, nonanalytic refusal, sample reuse, and finite-statistic
certification checks. Non-finite weighted inputs, statistics, normalization,
coefficients, and reconstructed spot values are refused; signed responses and
finite zero statistics remain valid. The exact-power
test uses 33 node realizations plus three checks, compared with the original
513 plus three.

The production standardized-Duchon integration gate (600 rows, 11 columns)
certifies at 129 nodes plus three checks: 132 design realizations instead of
516. Tensor construction takes 1.799842 seconds; the complete test takes 2.53
seconds. Its unchanged coefficient checks against an independently streamed
solve pass with zero error. Both moved probes rotate the reduced basis, so the
existing projector witness requires exact streamed evaluation there. This
checks preservation of that routing safeguard; it does not measure tensor
approximation error on an admitted moved-coordinate fast path.

The original 50,000-row fixture's repaired fit time and objective remain
unverified. These smaller gates are not an end-to-end reproduction of it.

The 16-test finite-guard source `psi_gram_tensor.rs` Git blob is
`6043c2ee4f8d9ca811feb504593e495e83cd78f2`.
The 600-row integration measurement used the preceding adaptive source blob
`3f9a9494131c13ce568f4329c902355ee3bb2480`, before the explicit finite-value guards.
