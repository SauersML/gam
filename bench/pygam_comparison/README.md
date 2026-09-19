# gamfit vs pyGAM measurements

This directory holds the data behind `docs/benchmarks.md`. Rebuild that page with
`python scripts/gen_benchmarks_doc.py`. `tests/test_benchmarks_doc.py` fails when the
committed page no longer matches this data.

| file | contents | source |
|---|---|---|
| `fit_cpu.tsv` | fit CPU seconds, median over `reps` | pyGAM audit `speed.md` §3.1 |
| `predict_cpu_ms.tsv` | predict CPU milliseconds for n new rows | pyGAM audit `speed.md` §3.3 |
| `peak_rss_mb.tsv` | peak RSS of the fitting process, MB | pyGAM audit `speed.md` §3.2 |
| `accuracy_cv.md` | 5-fold CV accuracy per case, verbatim | pyGAM audit `accuracy/partial.txt` |
| `tour_wall_s.tsv` | wall seconds for the real-data tour examples | pyGAM audit `docs.md`, tour summary |
| `reference_quality_pygam.tsv` | the pyGAM rows of one reference-quality CI run | `bench/gha_results/reference-quality/quality_results.tsv` |
| `reference_quality_run.json` | metadata of that run | `bench/gha_results/reference-quality/_run.json` |
| `audit_versions.json` | gamfit and pyGAM versions the audit measured | the audit's README |

The pyGAM audit is at
<https://github.com/SauersML/gam/tree/4a84d11a5b37b705551f758f473eb68fe08bf8b0/bench/pygam_audit>.
Its measurements used:

- gamfit wheel 0.1.267 and pygam 0.12.0, with numpy 2.4.6, pandas 3.0.6 and Python 3.11;
- a shared 4-CPU machine with a 1-minute load average of 33-48;
- one thread per library (`RAYON_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1`);
- a fresh subprocess per measurement.

CPU time is the primary speed metric because of that load. The raw speed records were not
committed, so the three speed tables are transcribed from the medians in `speed.md`. Wherever
a table has `-`, that cell was not measured.

The reference-quality workflow republishes `bench/gha_results/reference-quality/` on every
scheduled run. The documentation reads a committed snapshot, so it does not change underneath
the test. To take a newer run, use `python scripts/gen_benchmarks_doc.py --refresh-reference-quality`.
