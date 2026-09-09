#!/usr/bin/env bash
# Sourced by every release lane, including maturin's Linux container shell.

release_cache_start() {
  local attempt
  # The daemon's idle timer measures time between requests, including while a
  # long rustc/LTO request is still running. Keep the receipt and client alive
  # until this job explicitly stops its daemon.
  export SCCACHE_IDLE_TIMEOUT=0
  for attempt in 1 2 3 4 5; do
    if sccache --start-server; then
      sccache --zero-stats || return $?
      export RUSTC_WRAPPER="$(command -v sccache)"
      return 0
    fi
    if (( attempt < 5 )); then
      echo "::warning::Compiler cache startup failed (attempt ${attempt}/5); retrying in $((20 * attempt)) seconds."
      sleep "$((20 * attempt))"
    fi
  done
  echo '::error::Compiler cache startup failed after five attempts.'
  return 1
}

release_cache_verify_receipt() {
  awk '
    BEGIN {
      required["Compile requests"] = 1
      required["Cache hits"] = 1
      required["Cache misses"] = 1
      required["Cache errors"] = 1
      required["Cache read errors"] = 1
      required["Cache write errors"] = 1
    }
    {
      sub(/\r$/, "")
      print
      key = $1
      for (i = 2; i < NF; i++) key = key " " $i
      if (key in required) {
        seen[key]++
        if ($NF !~ /^[0-9]+$/) invalid = 1
        value[key] = $NF + 0
      }
    }
    END {
      for (key in required) if (seen[key] != 1) invalid = 1
      if (invalid) {
        print "::error::Incomplete or invalid sccache receipt; refusing an unmeasured release build."
        exit 1
      }
      if (value["Compile requests"] == 0) {
        print "::error::No compiler requests reached this sccache daemon; check wrapper binding and daemon lifetime."
        exit 1
      }
      # A READ error can hand the compiler a wrong object; a WRITE error is
      # the backend refusing to STORE an object the compiler already produced,
      # and the artifact is what it would be with no cache at all. Measured
      # 2026-09-09: Publish to PyPI 34376153113 refused three wheels whose
      # 385-390 compile requests all executed, on 80/120/387 write errors
      # from the shared backend, and Build and Release All 34373507539 the
      # same on 5. Cache history must not decide whether an artifact can be
      # published (pypi-wheels.yml), so a write error is reported and the
      # unseeded cache is named, but the receipt stands.
      errors = value["Cache errors"] + value["Cache read errors"]
      if (errors != 0) {
        print "::error::Compiler cache reported " errors " read error(s); refusing an unhealthy cache receipt."
        exit 1
      }
      if (value["Cache write errors"] != 0) {
        print "::warning::Compiler cache refused " value["Cache write errors"] " write(s): this build compiled every object itself and did not seed the cache; the artifact is unaffected."
      }
      if (value["Cache hits"] == 0) {
        print "::notice::Cold compiler cache: this successful build seeds subsequent releases."
      }
      print "Measured release build: requests=" value["Compile requests"] ", hits=" value["Cache hits"] ", misses=" value["Cache misses"] ", cache errors=0."
    }
  '
}

release_cache_verify() {
  local stats
  stats=$(sccache --show-stats) || return $?
  release_cache_verify_receipt <<< "$stats"
}
