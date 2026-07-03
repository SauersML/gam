#!/bin/bash
# Build and install the current MSI gamfit wheel from origin/main.
# CPU-only: submit on msismall; this does not request or launch GPU work.
# Submit from MSI:
#   cd /projects/standard/hsiehph/sauer354
#   sbatch --parsable msi_build_gamfit_current.sbatch
# The job writes /projects/standard/hsiehph/sauer354/gamfit_current_manifest.sh
# with the exact commit, wheel path, and venv path it installed.
#SBATCH --job-name=gamfit_current
#SBATCH --partition=msismall
#SBATCH --account=hsiehph
#SBATCH --cpus-per-task=24
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/build_gamfit_current_%j.out
set -euo pipefail

ROOT=/projects/standard/hsiehph/sauer354
REMOTE_URL=https://github.com/SauersML/gam.git
COMMIT=$(git ls-remote "$REMOTE_URL" refs/heads/main | awk '{print $1}')
if [ -z "$COMMIT" ]; then
  echo "could not resolve origin/main from $REMOTE_URL" >&2
  exit 79
fi
SHORT=${COMMIT:0:8}
BUILD=$ROOT/gam_wheel_$SHORT
WHEELDIR=$ROOT/wheels_$SHORT
BUILD_VENV=$ROOT/gamfit-build-$SHORT-venv
MANIFEST=$ROOT/gamfit_current_manifest.sh
PY=/usr/bin/python3.11

for path in "$BUILD" "$WHEELDIR" "$BUILD_VENV"; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite existing path: $path" >&2
    exit 80
  fi
done

export HOME=$ROOT
export XDG_CACHE_HOME=$ROOT/.cache
export XDG_CONFIG_HOME=$ROOT/.config
export CARGO_HOME=$ROOT/.cargo
export RUSTUP_HOME=$ROOT/.rustup
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_TARGET_DIR=$ROOT/scratch/target_gamfit_$SHORT
export CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-24}
export CARGO_INCREMENTAL=1
export CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
export RUST_BACKTRACE=1

module load openblas/0.3.27_gcc11.3.0_multiarch
OBLAS=$(printf '%s' "${LIBRARY_PATH:-}" | tr ':' '\n' | grep '/openblas/' | head -1)
if [ -z "$OBLAS" ] || [ ! -r "$OBLAS/libopenblas.so" ]; then
  echo "openblas module did not expose a readable libopenblas.so in LIBRARY_PATH" >&2
  echo "LIBRARY_PATH=${LIBRARY_PATH:-}" >&2
  exit 78
fi

echo "=== build metadata ==="
date -Is
echo "host=$(hostname)"
echo "commit=$COMMIT"
echo "python=$("$PY" --version 2>&1)"
echo "rustc=$(rustc --version)"
echo "cargo=$(cargo --version)"
echo "openblas=$OBLAS"

git clone --no-checkout "$REMOTE_URL" "$BUILD"
cd "$BUILD"
git fetch --depth 1 origin "$COMMIT"
git checkout --detach "$COMMIT"
ACTUAL=$(git rev-parse HEAD)
if [ "$ACTUAL" != "$COMMIT" ]; then
  echo "checked out $ACTUAL, expected $COMMIT" >&2
  exit 81
fi
git log --oneline -1

PROJECT_VERSION=$(awk -F\" '/^version = / {print $2; exit}' pyproject.toml)
CRATE_VERSION=$(awk -F\" '/^version = / {print $2; exit}' crates/gam-pyffi/Cargo.toml)
VERSION=$PROJECT_VERSION
if [ -z "$PROJECT_VERSION" ] || [ "$CRATE_VERSION" != "$PROJECT_VERSION" ]; then
  echo "version mismatch: pyproject=$PROJECT_VERSION crate=$CRATE_VERSION" >&2
  exit 82
fi
VENV=$ROOT/gamfit-$VERSION-$SHORT-venv
if [ -e "$VENV" ]; then
  echo "refusing to overwrite existing path: $VENV" >&2
  exit 80
fi

"$PY" -m venv "$BUILD_VENV"
"$BUILD_VENV/bin/python" -m pip install --upgrade pip wheel 'maturin>=1.14,<2'
mkdir -p "$WHEELDIR"
"$BUILD_VENV/bin/python" -m maturin build \
  --release \
  --compatibility manylinux_2_28 \
  --auditwheel warn \
  --interpreter "$PY" \
  --manifest-path "$BUILD/crates/gam-pyffi/Cargo.toml" \
  --out "$WHEELDIR"

WHEEL=$(find "$WHEELDIR" -maxdepth 1 -type f -name "gamfit-$VERSION-cp310-abi3-manylinux_2_28_x86_64.whl" | sort | head -1)
if [ -z "$WHEEL" ]; then
  echo "expected wheel not found in $WHEELDIR" >&2
  find "$WHEELDIR" -maxdepth 1 -type f -print >&2
  exit 83
fi
sha256sum "$WHEEL" | tee "$WHEELDIR/SHA256SUMS"
printf '%s\n' "$COMMIT" > "$WHEELDIR/SOURCE_COMMIT"

"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install "$WHEEL"
printf '%s\n' "$COMMIT" > "$VENV/GAMFIT_SOURCE_COMMIT"
printf '%s\n' "$WHEEL" > "$VENV/GAMFIT_WHEEL_PATH"
sha256sum "$WHEEL" | awk '{print $1}' > "$VENV/GAMFIT_WHEEL_SHA256"
WHEEL_SHA=$(cat "$VENV/GAMFIT_WHEEL_SHA256")

"$VENV/bin/python" - <<PY
import pathlib
import sys

import gamfit
import gamfit._rust as rust

expected_version = "$VERSION"
expected_commit = "$COMMIT"
venv = pathlib.Path("$VENV").resolve()
commit_marker = (venv / "GAMFIT_SOURCE_COMMIT").read_text().strip()
if gamfit.__version__ != expected_version:
    raise SystemExit(f"version {gamfit.__version__!r} != {expected_version!r}")
if commit_marker != expected_commit:
    raise SystemExit(f"commit marker {commit_marker!r} != {expected_commit!r}")
for imported_path in (
    pathlib.Path(gamfit.__file__).resolve(),
    pathlib.Path(rust.__file__).resolve(),
):
    if venv not in imported_path.parents:
        raise SystemExit(f"{imported_path} is outside {venv}")

print("VERIFY_OK")
print("python", sys.executable)
print("version", gamfit.__version__)
print("gamfit", gamfit.__file__)
print("rust", rust.__file__)
print("build_info", gamfit.build_info())
print("cuda_diagnostics", gamfit.cuda_diagnostics())
PY

cat > "$MANIFEST.tmp" <<MANIFEST
GAMFIT_MSI_VERSION=$VERSION
GAMFIT_MSI_COMMIT=$COMMIT
GAMFIT_MSI_SHORT=$SHORT
GAMFIT_MSI_VENV=$VENV
GAMFIT_MSI_WHEEL=$WHEEL
GAMFIT_MSI_PYTHON=$VENV/bin/python
GAMFIT_MSI_WHEEL_SHA256=$WHEEL_SHA
MANIFEST
mv "$MANIFEST.tmp" "$MANIFEST"

cat > "$ROOT/gamfit_0247_env.sh" <<'RETIRED'
#!/usr/bin/env bash
echo "FATAL gamfit MSI cutover: gamfit_0247_env.sh is retired." >&2
echo "Use /projects/standard/hsiehph/sauer354/gamfit_current_env.sh." >&2
case "$-" in
  *i*) return 87 ;;
  *) exit 87 ;;
esac
RETIRED

echo "WHEEL=$WHEEL"
echo "VENV=$VENV"
echo "COMMIT=$COMMIT"
echo "MANIFEST=$MANIFEST"
echo "=== done $(date -Is) ==="
