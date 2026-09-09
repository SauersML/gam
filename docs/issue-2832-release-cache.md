# Release cache gate and daemon lifetime: issues #2832 and #1267

Investigation on 2026-09-07 found two independent failure mechanisms in the
[failed wheel run 33654645604](https://github.com/SauersML/gam/actions/runs/33654645604).

1. Cache hits depend on cache history. A successful build with positive compiler
   requests, cold misses, and no cache errors is a valid release and a cache seed.
   Rejecting that build cannot establish artifact correctness or wrapper wiring.
2. The actual macOS ARM and musllinux ARM receipts in this run were entirely zero,
   including **Compile requests** and **Cache misses**. They do not demonstrate a
   cold cache. Merely changing the predicate from hits to requests would still
   reject these runs.

The daemon lifetime explains how measurements can disappear during long release
builds. In [sccache v0.17.0 server.rs](https://github.com/mozilla/sccache/blob/v0.17.0/src/server.rs),
`DEFAULT_IDLE_TIMEOUT` is 600 seconds. `ShutdownOrInactive::poll` restarts this timer
when a request arrives, without tracking its completion. `SccacheServer::run`
waits at most another 10 seconds for active requests after shutdown begins. A long
compiler request can therefore lose its daemon, and a long link can leave the
later statistics command starting a new, empty daemon. The documented
[`SCCACHE_IDLE_TIMEOUT=0`](https://github.com/mozilla/sccache/blob/v0.17.0/docs/Configuration.md)
disables this timer.

This mechanism is consistent with the Windows failure in the same run: the
`gam_models` compile lost its sccache response connection with os error 10054.
The log alone does not establish that every Windows connection reset has this
cause; a successful Windows release remains the end-to-end check for #1267.

There is also direct evidence for bounded cache startup retry: Linux x86_64 failed
at 2026-09-02T16:33:25 with `Unexpected (temporary)`, HTTP 503, `ServerBusy`, and
`Egress is over the account limit`. The startup policy makes five attempts with
20/40/60/80-second waits; exhausted startup still fails before compilation.

## Change

Both wheel and CLI release workflows now use one shared receipt implementation.
It requires exactly one numeric total for compiler requests, cache hits, cache
misses, and each of the three cache error counters. Missing, malformed, duplicate,
zero-request, and unhealthy receipts fail. Healthy zero-hit receipts emit a cold
cache notice and pass. Statistics command failures and counter reset failures are
also fatal.

The workflows disable idle shutdown before daemon startup, bind Cargo to the
actual installed wrapper, and explicitly stop each job's daemon. Docker startup,
measurement, and cleanup run in the same container as compilation. Native Windows
Cargo receives a Windows path. The
[sccache-action post hook](https://github.com/mozilla-actions/sccache-action/blob/9e7fa8a12102821edf02ca5dbea1acd0f89a2696/src/show_stats.ts)
is disabled because otherwise it starts an empty daemon after our cleanup to print
fresh zero counters. The real receipt remains in the build log.

The CLI workflow retains its warm dependency artifact cache. Swatinem's default
excludes workspace crate artifacts, so each fresh release runner still invokes
the compiler for workspace crates even when external dependencies are restored.

## Verification

On MSI acn112, in the existing validation repository, pinned to one CPU (8):

```sh
taskset -c 8 python3.12 -m unittest discover -s scripts/tests -p test_release_cache.py -v
```

All nine tests passed in 0.295 seconds. They cover cold and warm receipts, Windows
CRLF output, per-language statistics, all error counters, missing/malformed/
duplicate counters, zero requests, failed statistics commands, startup retry
success/exhaustion, and counter reset failure. Both release workflows execute this
suite before their build matrices. Ruby YAML parsing passed for both edited
workflows, and `bash -n` passed for the shared helper, also on MSI. No code, build,
or tests ran locally. A complete cross-platform release was not dispatched as
part of these focused checks.

## Amendment 2026-09-09: write errors no longer refuse a measured build

The receipt gate refused every wheel of the v0.3.156 matrix
([Publish to PyPI 34376153113](https://github.com/SauersML/gam/actions/runs/34376153113)):
the macOS-arm, musllinux-x86_64 and manylinux-arm jobs each executed 385–390
compile requests with zero compile failures and were refused on 387, 120 and 80
**write** errors from the shared cache backend; the same day
[Build and Release All 34373507539](https://github.com/SauersML/gam/actions/runs/34373507539)
was refused on 5. A write error is the backend declining to store an object the
compiler has already produced; the artifact is bit-for-bit what a build with no
cache would produce. Refusing it made cache-backend availability decide
publication, which is the premise this workflow rejects. Write errors are now
reported as a warning that names the unseeded cache. Read errors and cache
errors, which can hand the compiler a wrong object, remain fatal, and the
zero-request and malformed-receipt refusals are unchanged.

