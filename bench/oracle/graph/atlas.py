"""What each VPD subcomponent of vpd4l does, and which ones a behavior uses (#2951 graph oracle input).

The atlas, built once from Pile validation text: per subcomponent c of a weight matrix W ~ sum_c U_c V_c, its activation
at a token is a_c = (x . V_c) |U_c| (the size of what it writes there, signed by its direction); the atlas keeps each
subcomponent's root mean square activation over the text, its strongest activations with the tokens before them, and,
for subcomponents that write the residual stream (o, down), the tokens its direction raises and lowers through the
unembedding (the logit lens). The behavior table runs a behavior's prompts and changed prompts and lists the
subcomponents VPD's causal importance says the model needs there, with where and what each does.

  atlas.py build [--rows 1024] [--seq 512]          writes ~/mpd-data/graph_oracle/atlas/vpd4l.json
  atlas.py show BEHAVIOR.json [--top 48]            prints the behavior's table as the oracle sees it
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "vpd_2951"))

import mech  # noqa: E402

ATLAS = Path.home() / "mpd-data/graph_oracle/atlas/vpd4l.json"
TABLES = Path.home() / "mpd-data/graph_oracle/atlas/tables"  # <behavior id>.json: scores() per behavior, its first KEPT
KEPT = 2048
CODES = {site: code for code, site in mech.SITES.items()}
WRITERS = ("o_proj", "down_proj")
TOP = 8  # strongest activations kept per subcomponent
BEFORE = 6  # tokens of context kept before each


def site_of(name: str) -> tuple[int, str]:
    """('h.2.attn.v_proj') -> (2, 'v_proj')."""
    _, layer, _, kind = name.split(".")
    return int(layer), kind


def part(layer: int, kind: str, i: int) -> str:
    return f"<p:{layer}.{CODES[kind]}.{i}>"


@lru_cache(None)
def model(device: str = "mps"):
    import vpd_model

    target = vpd_model.load_target(device)
    return target, vpd_model.load_vpd(target, device)


def activations(ids: torch.Tensor, device: str = "mps") -> dict[str, torch.Tensor]:
    """Every subcomponent's activation a_c [B, T, C] per site name for token ids [B, T]."""
    target, vpd = model(device)
    vpd.clear()
    for n in vpd.names:
        target.site(n).cache_input = True
    with torch.no_grad():
        target(ids.to(device))
        out = {}
        for n in vpd.names:
            st = target.site(n)
            out[n] = (st.last_input @ st.V) * st.U.norm(dim=1)
    vpd.clear()
    return out


def build(rows: int, seq: int, batch: int, device: str) -> dict:
    import vpd_model

    target, vpd = model(device)
    tk = mech.tokenizer("vpd4l")
    ids = vpd_model.val_tokens(rows, seq)
    squares = {n: torch.zeros(vpd.C[n], dtype=torch.float64) for n in vpd.names}
    best = {n: (torch.zeros(0, vpd.C[n], device=device), torch.zeros(0, vpd.C[n], dtype=torch.long, device=device)) for n in vpd.names}
    for r in range(0, rows, batch):
        acts = activations(ids[r:r + batch], device)
        for n, a in acts.items():
            flat = a.reshape(-1, a.shape[-1])
            squares[n] += (flat ** 2).sum(0).cpu().double()
            where = torch.arange(flat.shape[0], device=device) + r * seq
            v, k = torch.cat([best[n][0], flat]), torch.cat([best[n][1], where[:, None].expand_as(flat)])
            top = v.abs().topk(min(TOP * 4, v.shape[0]), dim=0).indices
            best[n] = (v.gather(0, top), k.gather(0, top))
        print(f"{r + batch}/{rows} rows", flush=True)
    flat_ids = ids.reshape(-1).numpy()
    lens = (target.wte * target.ln_f).T  # [d, vocab]: the logit lens of a residual direction
    out = {"rows": rows, "seq": seq, "source": str(vpd_model.VAL_PARQUET.parent), "parts": {}}
    for n in vpd.names:
        layer, kind = site_of(n)
        rms = (squares[n] / (rows * seq)).sqrt().float().cpu()
        values, where = (t.cpu() for t in best[n])
        U = target.site(n).U
        if kind in WRITERS:  # top tokens raised and lowered, in chunks of subcomponents
            unit = U / U.norm(dim=1, keepdim=True)
            chunks = [unit[i:i + 256] @ lens for i in range(0, unit.shape[0], 256)]
            raised = torch.cat([c.topk(5, dim=1).indices for c in chunks]).tolist()
            lowered = torch.cat([(-c).topk(5, dim=1).indices for c in chunks]).tolist()
        for c in range(vpd.C[n]):
            seen, top = set(), []
            for v, w in zip(values[:, c].tolist(), where[:, c].tolist()):
                row, t = divmod(w, seq)
                key = (int(flat_ids[w]), v > 0)
                if key in seen:  # one example per token and sign
                    continue
                seen.add(key)
                start = row * seq + max(0, t - BEFORE)
                top.append([tk.decode(flat_ids[start:w].tolist()), tk.decode([int(flat_ids[w])]), round(v / max(rms[c].item(), 1e-12), 1)])
                if len(top) == TOP:
                    break
            entry = {"rms": round(rms[c].item(), 5), "top": top}
            if kind in WRITERS:
                entry["raises"] = [tk.decode([i]) for i in raised[c]]
                entry["lowers"] = [tk.decode([i]) for i in lowered[c]]
            out["parts"][part(layer, kind, c)] = entry
    return out


@lru_cache(None)
def load(path: Path = ATLAS) -> dict:
    return json.loads(path.read_text())


def describe(entry: dict, examples: int = 3) -> str:
    """One line: where the subcomponent fires most in text, and, for a residual writer, its logit lens."""
    fires = ", ".join(f"{(before[-24:] + '[' + tok + ']')!r} {z:+.0f}" for before, tok, z in entry["top"][:examples])
    line = f"fires on {fires}"
    if "raises" in entry:
        line += f"; raises {', '.join(map(repr, entry['raises'][:3]))}; lowers {', '.join(map(repr, entry['lowers'][:3]))}"
    return line


def sequences(behavior: dict) -> list[tuple[list[int], list[int], list[int] | None]]:
    """(prompt ids up to its last target, target positions, the changed prompt's ids as long or None) per prompt; what
    follows the last target is cut, since VPD's importance network reads later tokens."""
    out = []
    for p in behavior["prompts"]:
        T = max(p["target_positions"]) + 1
        cf = p.get("counterfactual")
        changed = cf["token_ids"][:T] if cf and len(cf["token_ids"]) >= T else None
        out.append((p["token_ids"][:T], p["target_positions"], changed))
    return out


def importance(ids: list[int], device: str = "mps") -> dict[str, torch.Tensor]:
    """VPD's causal importance [T, C] per site name (clamped to [0, 1]) on one sequence."""
    _, vpd = model(device)
    _, ci = vpd.target_and_ci(torch.tensor([ids], device=device))
    return {n: c[0].clamp(0, 1) for n, c in ci.items()}


def scores(behavior: dict, device: str = "mps", prompts: int = 32) -> dict[str, dict]:
    """Per subcomponent, means over the behavior's first `prompts` prompts: `need`, its causal importance summed over
    the positions up to the last target (VPD's measure of how much of it the model needs there); `moved`, the largest
    change of its activation between prompt and changed prompt at any position, in units of its typical activation in
    text; `moved_need`, that change weighted by the larger importance of the two; and `moved_at`, where `moved_need`
    peaks most often ("target" or the token there)."""
    rms = {n: torch.tensor([load()["parts"][part(*site_of(n), c)]["rms"] for c in range(C)], device=device).clamp_min(1e-12)
           for n, C in model(device)[1].C.items()}
    tk = mech.tokenizer(behavior["model"])
    sums: dict[str, dict[str, torch.Tensor]] = {}
    shifts: dict[str, list[dict]] = {}
    seqs = sequences(behavior)[:prompts]
    for ids, targets, changed in seqs:
        ci = importance(ids, device)
        a = activations(torch.tensor([ids]), device)
        b = activations(torch.tensor([changed]), device) if changed else None
        cb = importance(changed, device) if changed else None
        for n, c in ci.items():
            s = sums.setdefault(n, {k: torch.zeros(c.shape[1], device=device) for k in ("need", "moved", "moved_need")})
            s["need"] += c.sum(0)
            if b is not None:
                moved = ((a[n][0] - b[n][0]) / rms[n]).abs()
                weighted = moved * torch.maximum(c, cb[n])
                s["moved"] += moved.max(0).values
                s["moved_need"] += weighted.max(0).values
                words = shifts.setdefault(n, [dict() for _ in range(c.shape[1])])
                for k, t in enumerate(weighted.argmax(0).tolist()):
                    w = "target" if t in targets else name(tk, ids[t])
                    words[k][w] = words[k].get(w, 0) + 1
    out = {}
    for n, s in sums.items():
        values = {k: (v / len(seqs)).tolist() for k, v in s.items()}
        for c in range(len(values["need"])):
            out[part(*site_of(n), c)] = {**{k: values[k][c] for k in values},
                                         "moved_at": max(shifts[n][c].items(), key=lambda kv: kv[1])[0] if n in shifts else ""}
    return out


def name(tk, token: int) -> str:
    """A token as text, or its vocabulary entry when it decodes to nothing (a special token)."""
    return tk.decode([token]) or tk.id_to_token(token)


def ranking(table: dict[str, dict]) -> list[str]:
    """Subcomponents by how far their activation moves between the prompts and the changed prompts where the model
    needs them (`moved_need`), most first: what runs alike on both carries nothing the changed prompt removes."""
    return sorted(table, key=lambda p: -table[p]["moved_need"])


def table(behavior: dict, device: str = "mps") -> dict[str, dict]:
    """scores() of the behavior, from TABLES when written there."""
    path = TABLES / f"{behavior['id']}.json"
    return json.loads(path.read_text()) if path.exists() else scores(behavior, device)


def text(behavior: dict, top: int = 48, more: int = 208, device: str = "mps") -> str:
    """The behavior's subcomponent table for the oracle's prompt: the first `top` with what each does, then the next
    `more` with how far and where they move."""
    table_ = table(behavior, device)
    atlas = load()["parts"]
    ranked = ranking(table_)
    lines = ["Subcomponents whose activation differs most between the prompts and the changed prompts where the model needs"
             " them (the largest difference, in units of the subcomponent's typical activation in text, and the token"
             " where it is largest; VPD's causal importance summed over positions; what it does in text):"]
    for p in ranked[:top]:
        s = table_[p]
        lines.append(f"  {p} moves {s['moved']:.1f} at {s['moved_at']!r}, need {s['need']:.1f}: {describe(atlas[p])}")
    if more:
        lines.append("Next (subcomponent, difference, token):")
        rest = [f"{p} {table_[p]['moved']:.1f} {table_[p]['moved_at']!r}" for p in ranked[top:top + more]]
        lines += ["  " + "; ".join(rest[i:i + 6]) for i in range(0, len(rest), 6)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--rows", type=int, default=1024)
    b.add_argument("--seq", type=int, default=512)
    b.add_argument("--batch", type=int, default=8)
    b.add_argument("--device", default="mps")
    b.add_argument("--out", type=Path, default=ATLAS)
    t = sub.add_parser("tables", help="write scores() of every behavior in a directory to TABLES")
    t.add_argument("behaviors", type=Path)
    t.add_argument("--device", default="mps")
    s = sub.add_parser("show")
    s.add_argument("behavior", type=Path)
    s.add_argument("--top", type=int, default=48)
    s.add_argument("--device", default="mps")
    a = ap.parse_args()
    if a.cmd == "build":
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(build(a.rows, a.seq, a.batch, a.device)))
        print(f"wrote {a.out}")
    elif a.cmd == "tables":
        TABLES.mkdir(parents=True, exist_ok=True)
        for path in sorted(a.behaviors.glob("*.json")):
            behavior = json.loads(path.read_text())
            out = TABLES / f"{behavior['id']}.json"
            if out.exists() and len(json.loads(out.read_text())) <= KEPT:
                continue
            table_ = scores(behavior, a.device)
            out.write_text(json.dumps({p: {k: round(v, 3) if isinstance(v, float) else v for k, v in table_[p].items()}
                                       for p in ranking(table_)[:KEPT]}))
            print(f"{behavior['id']}", flush=True)
    else:
        print(text(json.loads(a.behavior.read_text()), a.top, a.device))


if __name__ == "__main__":
    main()
