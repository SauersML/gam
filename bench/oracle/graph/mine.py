"""A library of mechanisms by compressing a corpus of explanation graphs (#2951 graph oracle).

Each graph is a set of edges in coordinates relative to its target: (writer subcomponent, writer offset, reader
subcomponent or "out", reader offset). A library entry is a set of such edges defined once; a graph holding all of them
writes one reference instead. With size counted in edges, an entry of s edges referenced by k graphs changes the
corpus's size by (s + k) - k s. Greedy: grow an entry from the most frequent edge, adding the edge that co-occurs in the
most of the entry's graphs while the saving grows; keep it if it saves anything; remove its edges from the graphs that
reference it; repeat until no entry saves. Entries are the corpus's recurring mechanisms in this sense only.

  mine.py GRAPHS_DIR [GRAPHS_DIR...] --out mined.json [--rewrite DIR]   DIR gets the teacher answers with "uses" and
                                                                       library.json (score.py's LIBRARY)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mech  # noqa: E402

TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
CODE = {v: k for k, v in mech.SITES.items()}


def token(layer, kind, idx):
    return f"<p:{layer}.{CODE[kind]}.{idx}>"


def relative_edges(py: Path) -> frozenset | None:
    task = json.loads((TEXTS / f"{py.stem.split('.')[0]}.json").read_text())
    t = task["prompts"][0]["target_positions"][0]
    ir = mech.trace_inline(py.read_text(), "vpd4l", task)
    if not ir["valid"]:
        return None
    g = ir["graph"]
    nd = [tuple(x) for x in g["nodes"]]
    es = {(token(nd[w][0], nd[w][1], nd[w][3]), nd[w][2] - t, token(nd[r][0], nd[r][1], nd[r][3]), nd[r][2] - t) for r, w in g["parents"]}
    es |= {(token(nd[w][0], nd[w][1], nd[w][3]), nd[w][2] - t, "out", 0) for w in g["out"]}
    return frozenset(es)


def mine(graphs: dict[str, set]) -> list[dict]:
    graphs = {k: set(v) for k, v in graphs.items()}
    library, spent = [], set()  # spent: seeds whose best entry saves nothing
    while True:
        freq = Counter(e for es in graphs.values() for e in es if e not in spent)
        if not freq or freq.most_common(1)[0][1] < 2:  # an entry referenced once never saves
            break
        seed, _ = freq.most_common(1)[0]
        entry, holders = {seed}, {k for k, es in graphs.items() if seed in es}

        def saving(s, k):
            return k * s - (s + k)

        best = saving(len(entry), len(holders))
        while True:
            co = Counter(e for k in holders for e in graphs[k] if e not in entry)
            cand = None
            for e, _ in co.most_common(64):
                h = {k for k in holders if e in graphs[k]}
                sv = saving(len(entry) + 1, len(h))
                if sv > best:
                    best, cand = sv, (e, h)
            if cand is None:
                break
            entry.add(cand[0])
            holders = cand[1]
        if best <= 0:
            spent.add(seed)
            continue
        library.append({"edges": sorted(entry), "graphs": sorted(holders), "saving": best})
        for k in holders:
            graphs[k] -= entry
    return library


def rewrite(lib: list[dict], dirs: list[Path], out: Path) -> None:
    """Each teacher answer with the library entries it holds as "uses" in place of their edges -> OUT/<stem>.py, and
    OUT/library.json (score.py's LIBRARY: {"entries": {name: {"edges": [...]}}})."""
    import native

    out.mkdir(parents=True, exist_ok=True)
    names = {i: f"m{i}" for i in range(len(lib))}
    holds = {}
    for i, e in enumerate(lib):
        for k in e["graphs"]:
            holds.setdefault(k, []).append(i)
    nat = native.Native()
    for d in dirs:
        for py in sorted(d.glob("*.py")):
            stem = py.stem
            task = json.loads((TEXTS / f"{stem.split('.')[0]}.json").read_text())
            t = task["prompts"][0]["target_positions"][0]
            ir = mech.trace_inline(py.read_text(), "vpd4l", task)
            if not ir["valid"]:
                continue
            g = nat.from_ir(ir)
            drop = {tuple(x) for i in holds.get(stem, []) for x in lib[i]["edges"]}

            def rel(nd):
                layer, kind = int(nd[0].split(".")[1]), nd[0].split(".")[-1]
                return token(layer, kind, nd[2]), nd[1] - t

            g.parents = {r: [w for w in ws if (*rel(w), *rel(r)) not in drop] for r, ws in g.parents.items()}
            g.parents = {r: ws for r, ws in g.parents.items() if ws}
            g.out = [w for w in g.out if (*rel(w), "out", 0) not in drop]
            g.nodes = {x for r, ws in g.parents.items() for x in [r, *ws]} | set(g.out) | {nd for nd in g.nodes if nd[0].split(".")[-1] in ("q_proj", "k_proj")}
            claims = [tuple(c) for c in ir["graph"].get("claims", [])]
            (out / f"{stem}.py").write_text(native.program(g, claims, [names[i] for i in holds.get(stem, [])]))
    (out / "library.json").write_text(json.dumps({"entries": {names[i]: {"edges": e["edges"], "graphs": len(e["graphs"])} for i, e in enumerate(lib)}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rewrite", type=Path, help="write the teacher answers with library references here, and library.json")
    a = ap.parse_args()
    graphs = {}
    for d in a.dirs:
        for py in sorted(d.glob("*.py")):
            es = relative_edges(py)
            if es:
                graphs[py.stem] = es
    total = sum(len(v) for v in graphs.values())
    lib = mine(graphs)
    after = total - sum(e["saving"] for e in lib)
    a.out.write_text(json.dumps({"graphs": len(graphs), "edges": total, "edges_after": after, "library": lib}, indent=1))
    print(f"{len(graphs)} graphs, {total} edges; {len(lib)} entries; corpus size {total} -> {after} ({after / total:.0%})")
    if a.rewrite:
        rewrite(lib, a.dirs, a.rewrite)
    for i, e in enumerate(lib[:12]):
        print(f"  entry {i}: {len(e['edges'])} edges in {len(e['graphs'])} graphs (saves {e['saving']}): " + ", ".join(f"{w}@{wo:+d}->{r}@{ro:+d}" for w, wo, r, ro in e["edges"][:4]) + (" ..." if len(e["edges"]) > 4 else ""))


if __name__ == "__main__":
    main()
