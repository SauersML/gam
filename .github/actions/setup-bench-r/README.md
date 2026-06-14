# setup-bench-r

Composite action for making `Rscript` available in benchmark jobs.

It first tries `r-lib/actions/setup-r` for the requested R version. If
that action does not complete successfully, it installs R from the CRAN
Ubuntu apt repository and then verifies `Rscript --version`.

Inputs:

| Input | Default | Meaning |
| --- | --- | --- |
| `r-version` | `4.4.3` | R version requested from `r-lib/actions/setup-r`. |

Package installation is handled by `../bench-setup`; this action only
guarantees that R itself is present.
