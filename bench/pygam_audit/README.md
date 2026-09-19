# gamfit vs pyGAM audit (2026-09)

Audit of gamfit 0.1.267 against pyGAM 0.12.0, covering eleven axes. Each `<axis>.md` report lists findings with file:line evidence, repro scripts, and a verdict: gap, bug, already-better, or pyGAM slop to avoid. The scripts that produced the numbers are in `<axis>/`.

Scripts were run from a scratch venv outside the repo (pip `gamfit==0.1.267`, `pygam==0.12.0`, sklearn, pandas). Absolute scratch paths in the scripts are left over from that environment; point them at your own venv.

The reports are working notes used to plan the follow-up PRs. They are not user documentation. Some reports still contain placeholder sections (e.g. `RESULTS_TABLE`) where the auditor had not yet filled in a table; the findings in them are still valid.

| Report | Axis |
|---|---|
| api.md | Python API, sklearn contract |
| packaging.md | wheels, import, fork safety, typing |
| robustness.md | degenerate data, error handling |
| speed.md | fit/predict time, memory, scaling |
| docs.md | docs and examples walkthrough |
| families.md | response families and links |
| inference.md | intervals, p-values, UQ coverage |
| slop.md | pyGAM methodology not to copy; slop inside gamfit |
| pygam_tests.md | pyGAM's test suite as an oracle |
| accuracy.md, terms.md | predictive accuracy; terms and bases (may be partial) |
