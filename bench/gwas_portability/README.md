# Real GWAS + P+T portability simulation

This track simulates custom `msprime` demographies, trains a PLINK2 GWAS
plus pruning-and-thresholding PGS in one training deme, and evaluates binary
and survival prediction portability on held-out same-deme and other-deme
tests.

MSI direct-node run:

```bash
bash bench/gwas_portability/run_msi_direct.sh
```

The default MSI output root is:

```text
/projects/standard/hsiehph/sauer354/gam_gwas_portability
```

The runner intentionally does not use `stdpopsim`, score warping,
interactions, `linkwiggle`, `interSpline`, `z*PC`, score-warping, or
time-wiggling. Accuracy metrics are pooled only. Calibration summaries are
stratified by deme and genetic distance and never include within-stratum
AUC or C-index.
