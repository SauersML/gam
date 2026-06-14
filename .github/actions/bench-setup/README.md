# bench-setup

Composite action for preparing benchmark dependencies in GitHub Actions.

It checks out the repository, installs Rust stable, sets up Python 3.11,
sets up R through `../setup-bench-r`, restores package/tool caches, and
optionally installs the Python and R packages needed by benchmark jobs.

Inputs:

| Input | Default | Meaning |
| --- | --- | --- |
| `install-runtime-deps` | `true` | Install Python and R benchmark packages in the job environment. |
| `r-version` | `4.4.3` | R version requested from the R setup action. |

Use this action from benchmark workflows instead of duplicating setup
steps in each job. If the dependency list changes, update the install
script in `action.yml` and this README together.
