"""Regression test for #1560 — the benchmark suite ran zero scenarios.

`.github/workflows/benchmark.yml` fans the selected scenarios across two jobs
and reads four outputs from the `prepare` job's matrix step:

    parallel_matrix / parallel_count   (bench-shard,        max-parallel 8)
    serial_matrix   / serial_count     (bench-shard-serial, max-parallel 1)

`build_matrix()` previously emitted only a single `matrix=` output (plus
`is_nightly=`), leaving all four of those downstream outputs empty. The shard
jobs then expanded `fromJSON('')` to nothing and ran **zero** scenarios, so the
nightly Benchmark Suite went green while benchmarking nothing.

These assertions pin the output contract the workflow actually consumes.
"""

import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW_TASKS = REPO_ROOT / ".github" / "scripts" / "workflow_tasks.py"
SCENARIOS = REPO_ROOT / "bench" / "scenarios.json"

# Serial scenarios are kept in lockstep with build_matrix()'s SERIAL_SCENARIOS.
SERIAL_SCENARIOS = {"icu_survival_death", "cirrhosis_survival"}


def _load_workflow_tasks():
    spec = importlib.util.spec_from_file_location("workflow_tasks", WORKFLOW_TASKS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_build_matrix(monkeypatch, tmp_path, event_name):
    mod = _load_workflow_tasks()
    out = tmp_path / "github_output"
    out.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_EVENT_NAME", event_name)
    # build_matrix() reads bench/scenarios.json relative to cwd.
    monkeypatch.chdir(REPO_ROOT)
    mod.build_matrix()
    parsed = {}
    for line in out.read_text().splitlines():
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        parsed[key] = value
    return parsed


@pytest.mark.parametrize("event_name", ["schedule", "workflow_dispatch"])
def test_build_matrix_emits_the_four_outputs_the_workflow_reads(
    monkeypatch, tmp_path, event_name
):
    outputs = _run_build_matrix(monkeypatch, tmp_path, event_name)

    # The exact four output names benchmark.yml's `prepare` job declares.
    for key in ("parallel_matrix", "parallel_count", "serial_matrix", "serial_count"):
        assert key in outputs, f"build_matrix() must emit `{key}` (consumed by benchmark.yml)"

    # Each matrix must be valid JSON of the shape `fromJSON(...)` expects.
    parallel = json.loads(outputs["parallel_matrix"])
    serial = json.loads(outputs["serial_matrix"])
    assert "scenario" in parallel and isinstance(parallel["scenario"], list)
    assert "scenario" in serial and isinstance(serial["scenario"], list)

    # Counts must match the matrices the shard jobs expand, and the `!= '0'`
    # gate must actually open: at least the parallel shards must have work.
    assert outputs["parallel_count"] == str(len(parallel["scenario"]))
    assert outputs["serial_count"] == str(len(serial["scenario"]))
    assert int(outputs["parallel_count"]) > 0, "no parallel scenarios scheduled — suite would run nothing"

    # Serial scenarios route to the serial (max-parallel 1) job, never the
    # parallel one; everything else is parallel. No scenario is dropped.
    for name in SERIAL_SCENARIOS:
        assert name not in parallel["scenario"]
    assert not (set(parallel["scenario"]) & set(serial["scenario"]))


def test_nightly_selects_every_scenario_split_across_both_jobs(monkeypatch, tmp_path):
    outputs = _run_build_matrix(monkeypatch, tmp_path, "schedule")
    all_names = {
        s["name"]
        for s in json.loads(SCENARIOS.read_text())["scenarios"]
        if "name" in s
    }
    scheduled = set(json.loads(outputs["parallel_matrix"])["scenario"]) | set(
        json.loads(outputs["serial_matrix"])["scenario"]
    )
    # Nightly runs the whole suite — nothing silently dropped.
    assert scheduled == all_names
    # Serial scenarios that exist in the suite land in the serial bucket.
    expected_serial = all_names & SERIAL_SCENARIOS
    assert set(json.loads(outputs["serial_matrix"])["scenario"]) == expected_serial
