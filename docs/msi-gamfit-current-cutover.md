# MSI gamfit current wheel cutover

The active MSI wheel venv is pinned by source commit, not just by package
version:

```bash
/projects/standard/hsiehph/sauer354/gamfit-0.1.247-f6ce7eea-venv
```

It is installed from:

```bash
/projects/standard/hsiehph/sauer354/wheels_f6ce7eea/gamfit-0.1.247-cp310-abi3-manylinux_2_28_x86_64.whl
```

The required source commit is:

```text
f6ce7eeac90fd182fa62a4dba3ffcf6736f835f9
```

Use this invocation for MSI wheel-backed Python work:

```bash
source /projects/standard/hsiehph/sauer354/gam_env.sh
source /projects/standard/hsiehph/sauer354/gamfit_current_env.sh
gamfit_assert_msi_current
gamfit_python -c 'import gamfit, gamfit._rust; print(gamfit.__version__, gamfit.__file__, gamfit._rust.__file__)'
```

If a script activates another environment, source `gamfit_current_env.sh` after
that activation. The guard fails non-interactive scripts when `gamfit` or
`gamfit._rust` resolves outside the commit-pinned venv, or when the venv's
`GAMFIT_SOURCE_COMMIT` marker does not match the required commit.

Do not source `saevenv`, `saevenv_head`, or `gamfit_0247_env.sh` for gamfit
runtime work. They are not commit-pinned to current `origin/main`.

To rebuild this cutover wheel on CPU resources only:

```bash
cd /projects/standard/hsiehph/sauer354
sbatch --parsable msi_build_gamfit_current.sbatch
```
