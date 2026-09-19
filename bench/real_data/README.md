# Real-data leaderboard: gamfit vs pyGAM

Cross-validated fits of gamfit and pyGAM on pyGAM's own datasets, the classic
R / mgcv examples and a set of UCI regression and classification problems.
The committed leaderboard is [`results/LEADERBOARD.md`](results/LEADERBOARD.md)
(machine-readable: `results/leaderboard.json`, raw per-fold records:
`results/records.jsonl`, run metadata: `results/meta.json`).

## Running

```
pip install -r bench/real_data/requirements.txt   # bench-only: pyGAM, pandas, xlrd, psutil
maturin develop --release                         # the gamfit under test
python -m bench.real_data.run --out bench/real_data/results
python -m bench.real_data.report bench/real_data/results   # re-render only
```

`--datasets a,b`, `--libs gamfit,pygam_gs` and `--folds 0,1` select a subset.

## Data

Nothing is vendored. `datasets.py` records, for every file, its public URL
and SHA-256; the first run downloads it into `$GAMFIT_REAL_DATA_CACHE`
(default `~/.cache/gamfit-bench/real_data`), verifies the checksum (a
mismatch is an error, never a silent re-download), and builds a model-ready
`.npz` under `prepared/`. `gamsim1` is generated from a fixed seed.

| group | datasets | source |
|---|---|---|
| pyGAM | mcycle, coal, faithful, wage, trees, default, cake, hepatitis, head_circumference, chicago, chicago_docs | `pygam/datasets/*.csv` at the pyGAM v0.12.0 tag, built as pyGAM's own loaders build them |
| R classic | airquality, co2 (Mauna Loa), boston, gamsim1 | Rdatasets CSVs; gamsim1 re-creates `mgcv::gamSim(1)` |
| UCI | bike_hour, california, abalone, concrete, wine_red, wine_white, adult | UCI ML repository archives; California from the StatLib file behind sklearn's loader |

Each dataset carries one model: a tuple of terms (`s`, `factor`, `te`) that
renders to both the gamfit formula and the equivalent pyGAM terms, so both
libraries fit the same model. Basis sizes are each library's default except
where pyGAM's documentation sets one (`chicago_docs`: `s(time, n_splines=200)`).

## Protocol

* **Libraries.** `gamfit` (the explicit formula), `gamfit_auto` (`y ~ .` on
  the same columns, factor columns passed as strings), `pygam` (pyGAM at its
  default smoothing, `lam = 0.6`) and `pygam_gs` (pyGAM's `gridsearch`, its
  own way of choosing smoothing: the fair comparator).
* **Split.** 5-fold CV with a fixed seed (`worker.SPLIT_SEED`); binomial
  folds are stratified on the response so rare-event data keeps its events in
  every training set.
* **Isolation.** Every (dataset, fold, lib) is its own subprocess, run by
  `bench.pygam_compare`'s supervisor with every BLAS / OpenMP / Rayon pool at
  one thread. The supervisor polls the process tree's RSS (peak MB) and
  enforces a timeout and memory cap that are a harness safety net, not a
  solver budget: a rep that trips one is recorded as `timeout` / `memcap`
  and the later folds as `not_run_after_*`, all counted against the library.
* **Certification.** A gamfit fit returns only from a converged, certified
  optimization; a refusal is an error and shows as a fold not finished.
  pyGAM's "did not converge" warnings are counted per dataset.
* **Metrics.** Fit wall and CPU time (one cold fit), predict time, process
  peak RSS, held-out mean unit deviance (prior-weighted for `hepatitis`),
  held-out RMSE, and for Gaussian datasets the held-out coverage and mean
  width of the 95% prediction interval for a new observation.
