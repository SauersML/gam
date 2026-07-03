#!/usr/bin/env bash
# MSI gamfit wheel cutover guard. Source this after any other Python venv
# activation that might put an older gamfit on PATH.

GAMFIT_0247_ROOT=/projects/standard/hsiehph/sauer354
GAMFIT_0247_VERSION=0.1.247
GAMFIT_0247_VENV=${GAMFIT_0247_ROOT}/gamfit-0.1.247-venv
GAMFIT_0247_WHEEL=${GAMFIT_0247_ROOT}/wheels_head2/gamfit-0.1.247-cp310-abi3-manylinux_2_28_x86_64.whl
GAMFIT_0247_PYTHON=${GAMFIT_0247_VENV}/bin/python

_gamfit_0247_die() {
  local code="${2:-87}"
  echo "FATAL gamfit cutover: $1" >&2
  case "$-" in
    *i*) return "$code" ;;
    *) exit "$code" ;;
  esac
}

gamfit_assert_0247() {
  local py="${1:-${GAMFIT_0247_PYTHON}}"
  if [ ! -x "$py" ]; then
    echo "missing executable Python: $py" >&2
    return 87
  fi

  (cd /tmp && "$py" - <<'PY')
import pathlib
import sys

expected = "0.1.247"
venv = pathlib.Path("/projects/standard/hsiehph/sauer354/gamfit-0.1.247-venv").resolve()

try:
    import gamfit
    import gamfit._rust as rust
except Exception as exc:
    raise SystemExit(f"cannot import gamfit/_rust from cutover venv: {type(exc).__name__}: {exc}")

version = getattr(gamfit, "__version__", None)
pkg = pathlib.Path(gamfit.__file__).resolve()
ext = pathlib.Path(rust.__file__).resolve()

if version != expected:
    raise SystemExit(f"gamfit version {version!r} != {expected!r} at {pkg}")
if venv not in pkg.parents:
    raise SystemExit(f"gamfit imported from {pkg}, not {venv}")
if venv not in ext.parents:
    raise SystemExit(f"gamfit._rust imported from {ext}, not {venv}")

print(f"gamfit {version} OK: {pkg}")
print(f"gamfit._rust OK: {ext}")
PY
}

gamfit_use_0247() {
  [ -f "$GAMFIT_0247_WHEEL" ] || {
    echo "missing wheel: $GAMFIT_0247_WHEEL" >&2
    return 87
  }
  [ -x "$GAMFIT_0247_PYTHON" ] || {
    echo "missing venv Python: $GAMFIT_0247_PYTHON" >&2
    return 87
  }

  export GAMFIT_REQUIRED_VERSION="$GAMFIT_0247_VERSION"
  export GAMFIT_WHEEL="$GAMFIT_0247_WHEEL"
  export GAMFIT_PYTHON="$GAMFIT_0247_PYTHON"
  export PYTHONNOUSERSITE=1
  export PATH="${GAMFIT_0247_VENV}/bin:${PATH}"

  gamfit_assert_0247 "$GAMFIT_0247_PYTHON" >/dev/null
}

gamfit_python() {
  (cd /tmp && "$GAMFIT_0247_PYTHON" "$@")
}

gamfit_use_0247 || _gamfit_0247_die "expected gamfit ${GAMFIT_0247_VERSION} wheel venv is not usable"
