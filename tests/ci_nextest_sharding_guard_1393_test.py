"""Regression guard for #1393 — oversized nextest targets sharded under cap.

Issue #1393: after #1146 folded ~700 per-file integration crates into a handful
of aggregator binaries, a few of those binaries hold hundreds of `#[test]`s
each (quality ~427, basis_smooth ~292, manifolds ~152). Run serially under
`--test-threads 1` (the peak-RSS guard), the SUM of the legitimately-slow
REML/PIRLS/joint-Newton and R/Python reference-comparison tests in one binary
overran the 1800s per-binary GNU `timeout`, which SIGKILLed the whole target
(exit 124/137) BEFORE the per-test summary printed — so the entire target
failed in bulk with NO attribution of which test was slow.

The fix (commit d74ef8af) splits each oversized binary into nextest
`--partition count:i/N` shards. `--partition` is a pure run-time test FILTER, so
the binary still links exactly once and each shard reuses it (no extra link, no
extra disk), while each shard's wall-clock stays under a cap and per-test
attribution is preserved.

This pure-source contract test (no Rust build) pins the load-bearing pieces of
that fix so a refactor cannot silently revert to the bulk-kill behavior:

  1. the per-test slow-timeout layer (.config/nextest.toml) still names the
     offending test on overrun;
  2. the workflow shards the known-oversized binaries via `--partition`;
  3. the partitioned shards carry a dedicated (larger) wall-clock cap.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_nextest_profile_terminates_and_names_slow_tests():
    cfg = _read(".config/nextest.toml")
    assert "[profile.ci]" in cfg, "ci nextest profile missing"
    # A single overrunning #[test] must be SIGKILLed and reported by name —
    # this is the finer-grained layer the sharding sits on top of.
    assert "slow-timeout" in cfg, "per-test slow-timeout removed (#1393)"
    assert "terminate-after" in cfg, "terminate-after removed; hangs would run unbounded (#1393)"
    # no-fail-fast preserves the no-fail-fast contract across all shards.
    assert "fail-fast = false" in cfg, "fail-fast must stay false so all shards report (#1393)"


def test_workflow_shards_oversized_binaries_via_partition():
    wf = _read(".github/workflows/test.yml")
    assert "partitions_for_binary" in wf, "per-binary partition map removed (#1393)"
    # The known-oversized binaries must each request >1 partition.
    for binary in ("quality", "basis_smooth", "manifolds"):
        assert f"{binary})" in wf, f"{binary} dropped from partition map (#1393)"
    # The shards must run through nextest's --partition count:i/N filter.
    assert "--partition" in wf, "--partition sharding removed (#1393)"
    assert 'count:' in wf, "partition uses count:i/N filter form (#1393)"


def test_partitioned_shards_have_dedicated_cap():
    wf = _read(".github/workflows/test.yml")
    assert "PARTITIONED_BINARY_TIMEOUT" in wf, (
        "partitioned shards lost their dedicated wall-clock cap (#1393)"
    )
    # The partitioned cap must exceed the single-binary cap (a partition can
    # draw several multi-minute tests at once); read both and compare.
    def _int_after(token: str) -> int:
        for line in wf.splitlines():
            stripped = line.strip()
            if stripped.startswith(token + "="):
                return int(stripped.split("=", 1)[1].split()[0])
        raise AssertionError(f"{token} not found in workflow (#1393)")

    single = _int_after("INTEGRATION_TEST_TIMEOUT")
    partitioned = _int_after("PARTITIONED_BINARY_TIMEOUT")
    assert partitioned > single, (
        f"partitioned cap ({partitioned}s) must exceed single-binary cap "
        f"({single}s) so a heavy shard finishes and reports per-test (#1393)"
    )
