# GWAS + P+T Portability Simulation

This track simulates custom `msprime` demographies, trains a PLINK2 GWAS plus
pruning-and-thresholding PGS in `deme_0`, and evaluates binary and survival
prediction portability on held-out same-deme and other-deme tests.

MSI direct-node run:

```bash
bash bench/gwas_portability/run_msi_direct.sh
```

The default MSI output root is:

```text
/projects/standard/hsiehph/sauer354/gam_gwas_portability
```

`run_msi_direct.sh` expects the repo at
`/projects/standard/hsiehph/sauer354/gam_gwas_portability/gam`. It creates a
Python venv if needed and installs `.[test]`, `msprime`, and `scipy`. `plink2`
must already be on `PATH`.

Main outputs per run:

- `run_config.json`
- `metrics_global_pooled.tsv`
- `calibration_by_deme_distance.tsv`
- `pgs_portability_by_distance.tsv`
- pooled accuracy and calibration PNGs
- per-demography `pt_thresholds.tsv`, `simulated_data_*.tsv`, and
  `predictions.tsv`

The runner intentionally does not use `stdpopsim`, score warping,
interactions, `linkwiggle`, `interSpline`, `z*PC`, score-warping, or
time-wiggling. Accuracy metrics are pooled only. Calibration summaries are
stratified by deme and genetic distance and never include within-stratum
AUC or C-index.
