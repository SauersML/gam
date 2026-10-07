"""End-to-end checks of the graph oracle's score (#2951) on vpd4l induction: the reference programs trace,
the oracle prompt renders, and the checker orders them the way every valid score must:
  full program (every piece declared, every edge listed): execution error ~0, the largest opaque cost;
  empty program < random irrelevant heads (total bits);
  search's best program (e2e/search.py's result for the behavior, when one exists) < empty and < random.
The hand-written program's place is a measurement, not a requirement: test_hand_terms prints it.

  GRAPH_CHECKER=<mpd_graph_2951 binary> ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/e2e/test_e2e.py
The checker tests are skipped when the binary or the behavior file is missing.
"""

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPD_MEM_GIB", "1")

import mech  # noqa: E402
import programs  # noqa: E402
import prompt  # noqa: E402
import run as e2e  # noqa: E402
import score  # noqa: E402

BEHAVIOR = Path.home() / "mpd-data/graph_oracle/behaviors/vpd4l/induction_random.words8.json"
needs_checker = pytest.mark.skipif(not (score.BINARY.exists() and BEHAVIOR.exists()),
                                   reason="no checker binary (GRAPH_CHECKER) or no behavior file")


def test_references_trace():
    for name, source in programs.references("vpd4l").items():
        ir = mech.trace_inline(source, "vpd4l")
        assert ir["valid"], (name, ir["error"])
        assert (ir["nodes"] == []) == (name == "empty"), name
    full = mech.trace_inline(programs.full("vpd4l"), "vpd4l")
    s = mech.shapes("vpd4l")
    assert len(full["nodes"]) == 2 * s["layers"]
    # every writer (embed, then each block in causal order) into every later block and the logits
    blocks = 2 * s["layers"]
    assert len(full["edges"]) == blocks * (blocks + 1) // 2 + blocks + 1


def test_random_heads_avoid_hand():
    used = programs.hand_heads("vpd4l")
    ir = mech.trace_inline(programs.random_heads("vpd4l", 3, 0), "vpd4l")
    drawn = {(n["pieces"][0]["layer"], n["pieces"][0]["index"]) for n in ir["nodes"]}
    assert len(drawn) == 3 and not drawn & used


@pytest.mark.skipif(not BEHAVIOR.exists(), reason="no behavior file")
def test_prompt_renders():
    text = prompt.render(json.loads(BEHAVIOR.read_text()))
    assert "Write the program." in text and "vpd4l" in text


@pytest.fixture(scope="module")
def scores():
    with score.Checker("vpd4l") as c:
        e2e.load_behavior(c, BEHAVIOR)
        return {name: c.score(source, experiments=16, seed=0, reader=False)
                for name, source in programs.references("vpd4l").items()}


@needs_checker
def test_all_valid(scores):
    for name, r in scores.items():
        assert r["valid"], (name, r["error"])


@needs_checker
def test_full_program_is_the_model(scores):
    full = scores["full"]
    assert full["exec_error_bits"] / full["N"] < 1e-6, full["per_family"]
    assert full["opaque_numbers"] > max(r["opaque_numbers"] for n, r in scores.items() if n != "full")
    assert full["total_bits"] > scores["empty"]["total_bits"]


@needs_checker
def test_empty_beats_random_heads(scores):
    assert scores["empty"]["total_bits"] < scores["random"]["total_bits"]


@needs_checker
def test_hand_terms(scores):
    for n, r in scores.items():
        print(n, {k: r[k] for k in ("total_bits", "exec_error_bits", "opaque_bits", "code_bits")})


def best_search_program() -> str | None:
    """The lowest-total program search.py found for BEHAVIOR (any mode), or None."""
    found = []
    for path in (Path.home() / "mpd-data/graph_oracle/runs/search").glob(f"{BEHAVIOR.stem}.*.json"):
        r = json.loads(path.read_text())
        if r.get("stand_in") is None:  # scored under the checker's default stand-ins, as `scores` is
            found.append((r["score"]["total_bits"], r["source"]))
    return min(found)[1] if found else None


@needs_checker
@pytest.mark.skipif(best_search_program() is None, reason="no search result for the behavior yet")
def test_search_beats_empty_and_random(scores):
    with score.Checker("vpd4l") as c:
        e2e.load_behavior(c, BEHAVIOR)
        best = c.score(best_search_program(), experiments=16, seed=0, reader=False)
    assert best["total_bits"] < scores["empty"]["total_bits"], (best["total_bits"], scores["empty"]["total_bits"])
    assert best["total_bits"] < scores["random"]["total_bits"], (best["total_bits"], scores["random"]["total_bits"])


class FakePool:
    """Scores a search program by its units: each declared head costs 1, a neuron 0.01; heads (1, 1)
    and (2, 4) and MLP 0's neurons below 768 save 5, 5 and 0.04 per neuron when declared."""

    def __init__(self):
        self.calls = 0

    def score(self, sources, experiments, seed):
        import search
        out = []
        for src in sources:
            ir = mech.trace_inline(src, "vpd4l")
            assert ir["valid"], ir["error"]
            total = 100.0
            for n in ir["nodes"]:
                p = n["pieces"][0]
                size = mech.shapes("vpd4l")["heads" if p["kind"] == "head" else "d_mlp"]
                idx = range(size) if p["index"] is None else [p["index"]] if isinstance(p["index"], int) else p["index"]
                if p["kind"] == "head":
                    total += 1 - 5 * ((p["layer"], idx[0]) in {(1, 1), (2, 4)})
                else:
                    total += sum(0.01 - 0.04 * (p["layer"] == 0 and i < 768) for i in idx)
            out.append({"total_bits": total, "exec_error_bits": total, "opaque_bits": 0.0})
        self.calls += len(sources)
        return out


def test_greedy_addition_and_removal_find_the_planted_units():
    import search
    for mode in ("addition", "removal"):
        found = search.greedy(FakePool(), "vpd4l", mode, 1, 0, 384, lambda m: None)
        units = set(found["units"])
        assert ("head", 1, 1) in units and ("head", 2, 4) in units, (mode, units)
        assert not any(u[0] == "head" and (u[1], u[2]) not in {(1, 1), (2, 4)} for u in units), (mode, units)
        mlp = [u for u in units if u[0] == "mlp"]
        assert mlp and all(u[1] == 0 and u[3] <= 768 for u in mlp), (mode, mlp)
        assert sum(u[3] - u[2] for u in mlp) == 768, (mode, mlp)


def test_search_programs_trace_and_names_round_trip():
    import search
    units = search.all_units("vpd4l")
    for u in units + [b for b, _ in search.pieces_of(("mlp", 2, 0, 3072), 384)]:
        assert search.unit_of(search.name(u)) == u
    ir = mech.trace_inline(search.source(units), "vpd4l")
    assert ir["valid"], ir["error"]
    # every unit, and every causal edge among them, embed and the logits
    sites = sorted(search.site(u) for u in units)
    edges = sum(1 + sum(t < s for t in sites) for s in sites) + 1 + len(units)
    assert len(ir["nodes"]) == len(units) and len(ir["edges"]) == edges
    blocks = search.pieces_of(("mlp", 0, 0, 3072), 384)
    assert len(blocks) == 15 and all(sum(r[3] - r[2] for r in rest) + b[3] - b[2] == 3072 for b, rest in blocks)


def test_reader_subsample_is_stratified_and_complete():
    import full_score
    items = [{"family": f, "id": i} for i, f in enumerate(["clean"] * 3 + ["swap"] * 50 + ["cut"] * 10)]
    sample = full_score.subsample(items, 12)
    families = [it["family"] for it in sample]
    assert len(sample) == 12 and families.count("clean") == 3 and families.count("cut") >= 4
    assert len({it["id"] for it in sample}) == 12
    assert full_score.subsample(items[:5], 12) == items[:5]


def test_table_reads_sweep_search_and_oracle(tmp_path):
    import table
    n = 2**24
    terms = lambda t: {"total_bits": t * n, "exec_error_bits": t * n, "opaque_bits": 0.0, "code_bits": 0.0, "N": n, "opaque_numbers": 0}
    (tmp_path / "sweep").mkdir()
    (tmp_path / "sweep" / "a.b.json").write_text(json.dumps({"behavior": "a.b", "programs": {"empty": terms(3.0), "hand": terms(2.0)}}))
    (tmp_path / "search").mkdir()
    (tmp_path / "search" / "a.b.addition_cf.json").write_text(json.dumps({"heldout": terms(1.5), "score": terms(1.4), "calls": 99,
                                                                          "stand_in": "counterfactual"}))
    (tmp_path / "oracle").mkdir()
    for run, t in (("r1", 1.2), ("r2", 1.1)):
        (tmp_path / "oracle" / f"a.b.{run}.json").write_text(json.dumps({"behavior": "a.b", "score": terms(t)}))
    (tmp_path / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"set": "heldout", "step": 0, "behavior": "a.b", "mean_bits": 3.5 * n, "best_bits": 2.5 * n},
        {"set": "heldout", "step": 4, "behavior": "a.b", "mean_bits": 2.2 * n, "best_bits": 1.3 * n},
        {"summary": {}, "step": 4}]))
    rows = table.collect(tmp_path / "sweep", [tmp_path / "search"], tmp_path / "oracle", [tmp_path / "eval.jsonl"])
    got = {r[1]: (r[3], r[11]) for r in rows}
    assert got == {"empty": ("3.0000", ""), "hand": ("2.0000", ""), "search addition_cf": ("1.5000", "99"), "oracle": ("1.1000", ""),
                   "oracle best of n (step 4)": ("1.3000", ""), "oracle mean (step 4)": ("2.2000", "")}


def test_vpd_units_take_the_strongest_subcomponents():
    import search
    search.RANKING.update({"0.c_fc": [53, 726, 1131, 5], "0.down_proj": [3257, 1149, 607, 9]})
    u = ("vpd", 0, 2, 3)
    assert search.unit_of(search.name(u)) == u and search.site(u) == 1
    ir = mech.trace_inline(search.source([("head", 2, 4), u]), "vpd4l")
    assert ir["valid"], ir["error"]
    (vpd,) = [n for n in ir["nodes"] if n["pieces"][0]["view"] == "vpd"]
    got = {p["kind"]: p["index"] for p in vpd["pieces"]}
    assert got == {"c_fc": [53, 726], "down_proj": [607, 1149, 3257]}


def test_search_main_writes_its_result(tmp_path, monkeypatch):
    """search.py end to end with the fake pool: the result file holds the units, both scores and the
    checker calls (a shadowed variable once lost every MATS result at this last step)."""
    import search
    behavior = tmp_path / "x.y.json"
    behavior.write_text(json.dumps({"id": "x.y", "model": "vpd4l", "prompts": []}))

    class Pool(FakePool):
        def __init__(self, *a, **k):
            super().__init__()

        def close(self):
            pass

    monkeypatch.setattr(search, "Pool", Pool)
    monkeypatch.setattr(e2e, "record", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["search.py", str(behavior), "--mode", "addition", "--min-neurons", "384",
                                      "--start", "h1_1", "--out", str(tmp_path / "out")])
    search.main()
    r = json.loads((tmp_path / "out" / "x.y.addition.json").read_text())
    assert "h1_1" in r["units"] and "h2_4" in r["units"] and r["calls"] > 0
    assert r["heldout"]["total_bits"] == r["score"]["total_bits"]


def test_prefix_search_takes_pieces_that_pay_only_together():
    """Two planted units that save nothing alone but 50 together: one-at-a-time addition stops at empty,
    the prefix of the units ranked by their single patch finds both."""
    import search

    class Pool(FakePool):
        def score(self, sources, experiments, seed):
            out = []
            for src in sources:
                ir = mech.trace_inline(src, "vpd4l")
                heads = {(n["pieces"][0]["layer"], n["pieces"][0]["index"]) for n in ir["nodes"] if n["pieces"][0]["kind"] == "head"}
                exec_ = 100.0 - 50.0 * ({(1, 1), (2, 4)} <= heads) - 0.5 * len(heads & {(1, 1), (2, 4)})
                total = exec_ + 2.0 * len(ir["nodes"])
                out.append({"total_bits": total, "exec_error_bits": exec_, "opaque_bits": total - exec_, "N": 1})
            self.calls += len(sources)
            return out

    units = [("head", l, h) for l in range(4) for h in range(6)]
    found = search.prefix_search(Pool(), "vpd4l", 1, 0, 3072, lambda m: None, units=units)
    assert set(found["units"]) == {("head", 1, 1), ("head", 2, 4)}, found["units"]
    addition = search.greedy(Pool(), "vpd4l", "addition", 1, 0, 3072, lambda m: None, start=[])
    assert addition["units"] == [] or len(addition["units"]) < 2
