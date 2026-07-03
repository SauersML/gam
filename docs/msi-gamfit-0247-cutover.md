# MSI gamfit 0.1.247 wheel cutover

The active MSI wheel venv is:

```bash
/projects/standard/hsiehph/sauer354/gamfit-0.1.247-venv
```

It is installed from:

```bash
/projects/standard/hsiehph/sauer354/wheels_head2/gamfit-0.1.247-cp310-abi3-manylinux_2_28_x86_64.whl
```

Use this exact invocation for wheel-backed Python work:

```bash
source /projects/standard/hsiehph/sauer354/gam_env.sh
source /projects/standard/hsiehph/sauer354/gamfit_0247_env.sh
gamfit_assert_0247
gamfit_python -c 'import gamfit, gamfit._rust; print(gamfit.__version__, gamfit.__file__, gamfit._rust.__file__)'
```

If a script activates another venv such as `refpy` or `saevenv`, source
`gamfit_0247_env.sh` after that activation. The guard fails non-interactive
scripts when `gamfit` or `gamfit._rust` resolves outside the 0.1.247 venv.
