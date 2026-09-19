#!/usr/bin/env bash
# fetch_build_cache.sh — warm-start a cold cloud session's Rust compile cache from
# the `session-build-cache` artifact that .github/workflows/build-cache.yml
# ("Build cache (session warm start)") uploads on every push to main.
#
# HOW A SESSION GETS THE URL (GitHub MCP tools, no gh/token needed):
#   1. mcp__github__actions_list  method=list_workflow_runs
#        owner=SauersML repo=gam resource_id=build-cache.yml
#        workflow_runs_filter={branch: main, status: completed}
#      take the newest run whose conclusion is "success", then
#      mcp__github__actions_list  method=list_workflow_run_artifacts
#        resource_id=<run id>   -> the id of artifact "session-build-cache"
#   2. mcp__github__actions_get   method=download_workflow_run_artifact
#        resource_id=<artifact id>  -> a signed URL, valid for ~1 minute
#   3. IMMEDIATELY run:   scripts/fetch_build_cache.sh '<signed url>'
#      then `eval` / paste the export lines it prints (also saved to
#      ~/.cache/gam-build-cache.env, so `. ~/.cache/gam-build-cache.env` works
#      in every later shell) and build as usual with ./build.sh.
#
# What it does: downloads the zip, installs the pinned sccache binary and the
# key-normalising wrapper (scripts/sccache-rustc-wrapper.sh) into ~/.local/bin,
# unpacks the cache into ~/.cache/sccache, and prints the env to use.
#
# Idempotent: re-running with a fresh URL for the SAME commit's cache leaves the
# unpacked cache alone (it only re-installs the binaries and re-prints the env);
# a newer artifact replaces the old cache wholesale. Fails loudly (non-zero exit,
# message on stderr) on any download, integrity or extraction problem, and never
# leaves a half-extracted cache behind (it unpacks into a temp dir, then renames).
#
# Paths matter: sccache keys include the working directory and absolute source
# paths, so hits need the repo at /home/user/gam, CARGO_HOME=/root/.cargo (the
# default for root) and the rust-toolchain.toml toolchain — exactly what the CI
# job reproduces. No RUSTFLAGS.
#
# Env overrides: GAM_BUILD_CACHE_DIR (default ~/.cache/sccache),
#                GAM_BUILD_CACHE_BIN (default ~/.local/bin).
set -euo pipefail

die() { echo "fetch_build_cache: ERROR: $*" >&2; exit 1; }
log() { echo "fetch_build_cache: $*" >&2; }

[[ $# -eq 1 && -n "$1" ]] || die "usage: $0 '<signed artifact download URL>'  (see header for how to get it)"
URL="$1"
for t in curl python3 tar; do command -v "$t" >/dev/null 2>&1 || die "'$t' is required but not installed"; done

CACHE_DIR="${GAM_BUILD_CACHE_DIR:-$HOME/.cache/sccache}"
BIN_DIR="${GAM_BUILD_CACHE_BIN:-$HOME/.local/bin}"
ENV_FILE="$HOME/.cache/gam-build-cache.env"
mkdir -p "$BIN_DIR" "$(dirname "$CACHE_DIR")"

WORK=$(mktemp -d "$(dirname "$CACHE_DIR")/.gam-build-cache.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
ZIP="$WORK/artifact.zip"

log "downloading artifact…"
start=$(date +%s)
curl -fsSL --retry 5 --retry-delay 2 --retry-all-errors --connect-timeout 30 \
  -o "$ZIP" "$URL" \
  || die "download failed. Signed artifact URLs expire after about a minute; get a fresh one with download_workflow_run_artifact and run this straight away."
[[ -s "$ZIP" ]] || die "downloaded file is empty"
log "downloaded $(du -h "$ZIP" | cut -f1) in $(( $(date +%s) - start ))s"

# Validate the zip and pull out the small members (binary, wrapper, manifest).
python3 - "$ZIP" "$WORK" <<'PY' || die "artifact is not a valid session-build-cache zip"
import sys, zipfile, shutil, os
zp, work = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zp) as z:
    names = set(z.namelist())
    need = {"manifest.txt", "sccache", "sccache-rustc-wrapper", "sccache-cache.tar"}
    missing = need - names
    if missing:
        sys.exit("missing members: %s (have: %s)" % (sorted(missing), sorted(names)[:10]))
    for n in ("manifest.txt", "sccache", "sccache-rustc-wrapper"):
        with z.open(n) as src, open(os.path.join(work, n), "wb") as dst:
            shutil.copyfileobj(src, dst)
PY

manifest_val() { sed -n "s/^$1=//p" "$WORK/manifest.txt" | head -n1; }
NEW_COMMIT=$(manifest_val commit)
[[ -n "$NEW_COMMIT" ]] || die "manifest.txt has no commit= line"

chmod +x "$WORK/sccache" "$WORK/sccache-rustc-wrapper"
"$WORK/sccache" --version >/dev/null 2>&1 || die "bundled sccache binary does not run on this host"

# Stop any running server BEFORE touching its binary or directory (it would keep
# serving the old cache from memory-mapped state and the old size budget).
if command -v sccache >/dev/null 2>&1; then sccache --stop-server >/dev/null 2>&1 || true; fi
[[ -x "$BIN_DIR/sccache" ]] && { "$BIN_DIR/sccache" --stop-server >/dev/null 2>&1 || true; }

install -m 0755 "$WORK/sccache" "$BIN_DIR/sccache"
install -m 0755 "$WORK/sccache-rustc-wrapper" "$BIN_DIR/sccache-rustc-wrapper"

OLD_COMMIT=""
[[ -f "$CACHE_DIR/.gam-build-cache-manifest" ]] && \
  OLD_COMMIT=$(sed -n 's/^commit=//p' "$CACHE_DIR/.gam-build-cache-manifest" | head -n1)

if [[ "$OLD_COMMIT" == "$NEW_COMMIT" ]]; then
  log "cache for commit $NEW_COMMIT already unpacked in $CACHE_DIR — keeping it"
else
  log "unpacking cache for commit $NEW_COMMIT…"
  STAGE="$WORK/cache"
  mkdir -p "$STAGE"
  # Stream the tar member straight out of the zip into tar: no second full copy.
  python3 - "$ZIP" <<'PY' | tar -x -C "$STAGE" -f - \
    || die "extracting the cache failed (disk full? df -h $(dirname "$CACHE_DIR"))"
import sys, zipfile, shutil
with zipfile.ZipFile(sys.argv[1]) as z, z.open("sccache-cache.tar") as src:
    shutil.copyfileobj(src, sys.stdout.buffer, 1 << 20)
PY
  rm -f "$ZIP"   # free the space before the swap
  cp "$WORK/manifest.txt" "$STAGE/.gam-build-cache-manifest"
  rm -rf "$CACHE_DIR"
  mv "$STAGE" "$CACHE_DIR"
fi
rm -f "$ZIP"

# Cache budget: at least 10G, and 1.5x what we unpacked so the first builds can
# add entries without sccache's LRU evicting the warm ones.
size_kb=$(du -sk "$CACHE_DIR" | cut -f1)
size_gb=$(( (size_kb * 3 / 2 + 1048575) / 1048576 ))
(( size_gb < 10 )) && size_gb=10

cat >"$ENV_FILE" <<EOF
export PATH="$BIN_DIR:\$PATH"
export RUSTC_WRAPPER="$BIN_DIR/sccache-rustc-wrapper"
export SCCACHE_DIR="$CACHE_DIR"
export SCCACHE_CACHE_SIZE="${size_gb}G"
EOF

log "installed $("$BIN_DIR/sccache" --version) ; cache $(du -sh "$CACHE_DIR" | cut -f1) from commit $NEW_COMMIT ($(manifest_val built_at))"
log "toolchain the cache was built with: $(manifest_val rustc_release)"
want=$(manifest_val rustc_release)
have=$(cd /home/user/gam 2>/dev/null && rustc -V 2>/dev/null || true)
if [[ -n "$want" && -n "$have" && "$have" != *"$want"* ]]; then
  log "WARNING: local toolchain is '$have' — cache hits need rustc $want"
fi
[[ "$(pwd -P)" == /home/user/gam* || -d /home/user/gam ]] || \
  log "WARNING: no /home/user/gam checkout — cache keys contain that path, expect misses elsewhere"
log "run these (saved in $ENV_FILE; '. $ENV_FILE' in later shells):"
cat "$ENV_FILE"
cat <<'EOF'
# optional, for a one-shot build of clean main (every crate from cache, no incremental):
# export GAM_USE_SCCACHE=1
EOF
