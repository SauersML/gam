Unpublished core shared-tangent investigation for #2627:

The formula rotation failure reproduces directly in `gam-models`, without Python. Fixed-smoothing coefficient rotation error is `2.568167900562912e-12`; after smoothing optimization it was `3.505728864431333`, matching CI. The two frames selected lambdas `[5.1407870380537735e-5, 1.5990200099082836]` and `[6.987048966447412e-5, 9.357622968840175e-14]`, with different REML scores.

The combined penalty was reclassified against its largest scaled eigenvalue on every evaluation. Positive strengths cannot change the null space of a PSD penalty sum. The experimental repair freezes the unscaled balanced penalty range and evaluates its determinant/inverse on that fixed space.

MSI now reports **4 passed, 1 failed, 0 ignored, 1,555 filtered out, 0.31 seconds** for the five core checks. The rank-invariance, analytic gradient/Hessian, isotropic/Fisher equivalence, and streamed Fisher-statistics checks pass. The formula regression now passes coefficients (`8.076839963511162e-8`) and predictions (`1.1441612046692029e-8`) but still fails smoothing-strength agreement: the second strength is `0.3148111551710103` versus `0.15579674387361528`; scores differ by about `3.53e-8`.

I am measuring the flat direction's gradient/Hessian and the design/penalty spectra before deciding whether this is unresolved convergence or identifiability. No existing regression has been weakened, and this experiment is not yet published to main. The broader issue remains open.
