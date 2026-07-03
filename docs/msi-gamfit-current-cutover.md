# MSI gamfit current wheel cutover

The active MSI wheel venv is pinned by the generated MSI manifest, not just by
package version:

```bash
<MSI_PROJECT_ROOT>/gamfit_current_manifest.sh
```

That manifest contains the exact `origin/main` commit, wheel path, venv path,
and wheel SHA installed by the latest CPU build job:

```bash
cat <MSI_PROJECT_ROOT>/gamfit_current_manifest.sh
```

Use this invocation for MSI wheel-backed Python work:

```bash
source <MSI_PROJECT_ROOT>/gam_env.sh
source <MSI_PROJECT_ROOT>/gamfit_current_env.sh
gamfit_assert_msi_current
gamfit_python -c 'import gamfit, gamfit._rust; print(gamfit.__version__, gamfit.__file__, gamfit._rust.__file__)'
```

If a script activates another environment, source `gamfit_current_env.sh` after
that activation. The guard fails non-interactive scripts when `gamfit` or
`gamfit._rust` resolves outside the commit-pinned venv, or when the venv's
`GAMFIT_SOURCE_COMMIT` or `GAMFIT_WHEEL_SHA256` markers do not match the
manifest.

Do not source `saevenv`, `saevenv_head`, or `gamfit_0247_env.sh` for gamfit
runtime work. They are not commit-pinned to current `origin/main`.

To rebuild this cutover wheel on CPU resources only:

```bash
cd <MSI_PROJECT_ROOT>
sbatch --parsable msi_build_gamfit_current.sbatch
```
