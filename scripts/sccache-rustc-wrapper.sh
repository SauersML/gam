#!/bin/sh
# sccache-rustc-wrapper.sh — RUSTC_WRAPPER that makes sccache cache keys portable
# between the build-cache CI job (.github/workflows/build-cache.yml) and the
# cloud sessions that download its cache (scripts/fetch_build_cache.sh).
#
# sccache folds EVERY `CARGO_*` variable in rustc's environment into the hash
# key (minus CARGO_MAKEFLAGS / CARGO_BUILD_JOBS / CARGO_REGISTRIES_* /
# CARGO_ENCODED_RUSTFLAGS). Cargo passes the caller's whole environment through
# to rustc, so host-specific cargo *configuration* such as CARGO_HTTP_CAINFO
# (set by the session proxy), CARGO_TARGET_DIR / CARGO_INCREMENTAL (exported by
# build.sh) or CARGO_TERM_COLOR would make every key differ between CI and a
# session and turn a warm cache into 100% misses. Those variables configure
# cargo itself, never rustc (cargo has already turned them into rustc
# arguments), so dropping them here cannot change compiler output. Variables
# cargo sets FOR rustc (CARGO_PKG_*, CARGO_MANIFEST_*, CARGO_CRATE_NAME,
# CARGO_PRIMARY_PACKAGE, CARGO_TARGET_TMPDIR, CARGO_BIN_*, CARGO_CFG_*, …) are
# kept because code can read them with env!().
#
# Stripping CARGO_INCREMENTAL also stops sccache from refusing to start under
# build.sh's default CARGO_INCREMENTAL=1: workspace crates, which cargo compiles
# with `-C incremental`, then run uncached (incremental keeps working), while
# every registry/git dependency is served from the cache.
#
# The sccache binary is taken from $GAM_SCCACHE_BIN, else the one installed next
# to this wrapper, else `sccache` on PATH.
for v in $(env | sed -n 's/^\(CARGO_[A-Za-z0-9_]*\)=.*/\1/p'); do
  case "$v" in
    CARGO_HOME|CARGO_TARGET_DIR|CARGO_INCREMENTAL|CARGO_HTTP_*|CARGO_NET_*|\
    CARGO_TERM_*|CARGO_BUILD_*|CARGO_CACHE_*|CARGO_REGISTRY_*|CARGO_PROFILE_*|\
    CARGO_ALIAS_*|CARGO_UNSTABLE_*|CARGO_LOG*|CARGO_FUTURE_INCOMPAT_*|\
    CARGO_RESOLVER_*|CARGO_INSTALL_*|CARGO_SOURCE_*|CARGO_PATCH_*)
      unset "$v" ;;
  esac
done
if [ -n "${GAM_SCCACHE_BIN:-}" ]; then
  exec "$GAM_SCCACHE_BIN" "$@"
fi
here=$(dirname "$0")
if [ -x "$here/sccache" ]; then
  exec "$here/sccache" "$@"
fi
exec sccache "$@"
