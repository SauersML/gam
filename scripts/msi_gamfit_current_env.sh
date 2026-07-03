#!/usr/bin/env bash
# MSI gamfit wheel cutover guard. Source this after any Python environment
# activation; it forces the current commit-pinned wheel venv to the front.

GAMFIT_MSI_ROOT=/projects/standard/hsiehph/sauer354
GAMFIT_MSI_VERSION=0.1.247
GAMFIT_MSI_COMMIT=f6ce7eeac90fd182fa62a4dba3ffcf6736f835f9
GAMFIT_MSI_SHORT=f6ce7eea
GAMFIT_MSI_VENV=${GAMFIT_MSI_ROOT}/gamfit-${GAMFIT_MSI_VERSION}-${GAMFIT_MSI_SHORT}-venv
GAMFIT_MSI_WHEEL=${GAMFIT_MSI_ROOT}/wheels_${GAMFIT_MSI_SHORT}/gamfit-${GAMFIT_MSI_VERSION}-cp310-abi3-manylinux_2_28_x86_64.whl
GAMFIT_MSI_PYTHON=${GAMFIT_MSI_VENV}/bin/python

_gamfit_msi_die() {
  local code="${2:-87}"
  echo "FATAL gamfit MSI cutover: $1" >&2
  case "$-" in
    *i*) return "$code" ;;
    *) exit "$code" ;;
  esac
}

gamfit_assert_msi_current() {
  local py="${1:-${GAMFIT_MSI_PYTHON}}"
  if [ ! -x "$py" ]; then
    echo "missing executable Python: $py" >&2
    return 87
  fi
  if [ ! -f "${GAMFIT_MSI_VENV}/GAMFIT_SOURCE_COMMIT" ]; then
    echo "missing source commit marker: ${GAMFIT_MSI_VENV}/GAMFIT_SOURCE_COMMIT" >&2
    return 87
  fi
  if [ ! -f "${GAMFIT_MSI_VENV}/GAMFIT_WHEEL_SHA256" ]; then
    echo "missing wheel sha marker: ${GAMFIT_MSI_VENV}/GAMFIT_WHEEL_SHA256" >&2
    return 87
  fi

  (cd /tmp && "$py" - <<PY)
import pathlib

expected_version = "${GAMFIT_MSI_VERSION}"
expected_commit = "${GAMFIT_MSI_COMMIT}"
venv = pathlib.Path("${GAMFIT_MSI_VENV}").resolve()

try:
    import gamfit
    import gamfit._rust as rust
except Exception as exc:
    raise SystemExit(f"cannot import gamfit/_rust from MSI cutover venv: {type(exc).__name__}: {exc}")

version = getattr(gamfit, "__version__", None)
pkg = pathlib.Path(gamfit.__file__).resolve()
ext = pathlib.Path(rust.__file__).resolve()
commit = (venv / "GAMFIT_SOURCE_COMMIT").read_text().strip()

if version != expected_version:
    raise SystemExit(f"gamfit version {version!r} != {expected_version!r} at {pkg}")
if commit != expected_commit:
    raise SystemExit(f"gamfit source commit {commit!r} != {expected_commit!r}")
if venv not in pkg.parents:
    raise SystemExit(f"gamfit imported from {pkg}, not {venv}")
if venv not in ext.parents:
    raise SystemExit(f"gamfit._rust imported from {ext}, not {venv}")

print(f"gamfit {version} OK at commit {commit}: {pkg}")
print(f"gamfit._rust OK: {ext}")
PY
}

gamfit_use_msi_current() {
  [ -f "$GAMFIT_MSI_WHEEL" ] || {
    echo "missing wheel: $GAMFIT_MSI_WHEEL" >&2
    return 87
  }
  [ -x "$GAMFIT_MSI_PYTHON" ] || {
    echo "missing venv Python: $GAMFIT_MSI_PYTHON" >&2
    return 87
  }

  export GAMFIT_REQUIRED_VERSION="$GAMFIT_MSI_VERSION"
  export GAMFIT_REQUIRED_COMMIT="$GAMFIT_MSI_COMMIT"
  export GAMFIT_WHEEL="$GAMFIT_MSI_WHEEL"
  export GAMFIT_PYTHON="$GAMFIT_MSI_PYTHON"
  export PYTHONNOUSERSITE=1
  export PATH="${GAMFIT_MSI_VENV}/bin:${PATH}"

  gamfit_assert_msi_current "$GAMFIT_MSI_PYTHON" >/dev/null
}

gamfit_python() {
  (cd /tmp && "$GAMFIT_MSI_PYTHON" "$@")
}

gamfit_use_msi_current || _gamfit_msi_die "expected gamfit ${GAMFIT_MSI_VERSION} ${GAMFIT_MSI_SHORT} wheel venv is not usable"
