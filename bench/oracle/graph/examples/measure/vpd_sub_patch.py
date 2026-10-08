"""Which VPD subcomponents of vpd4l carry a behavior's answer, and how many of each block pay for
themselves (for R2's VPD-view programs). For every subcomponent i at every site, the counterfactual
run gets i's clean contribution in place of its own ((x_clean . v_i - x . v_i) u_i added at the site,
every position) and recovery = KL(M_clean || M_cf) - KL(M_clean || M_patched), bits per target token;
measured for the CANDIDATES subcomponents per site whose contribution changes most between the two
runs (sum over positions of |(x_clean - x_cf) . v_i| x |u_i|; the others are recorded as 0).
Then per block (a layer's attention: q/k/v/o_proj; its MLP: c_fc/down_proj) the top-k subcomponents
by recovery are patched together for k = 1, 2, 4, ..., and the k minimizing KL + their opaque price
((d_in + d_out) weights x 1/2 log2 N bits / N per token, N = 2^24) is the block's best k.
With --library, the same for decomp's library parts (each part's slices patched together; programs
vpd4l_<behavior>_lib.py). With --programs DIR, writes DIR/vpd4l_<behavior>_vpd.py: one node per block with a paying k, every
causal edge among them (embed into each, each into the logits), and a docstring of the measurements.

  MPD_MEM_GIB=6 mem-lease 6 ~/mpd-data/venv/bin/python vpd_sub_patch.py OUT_DIR BEHAVIOR.json [...] [--programs DIR]
"""

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # bench/oracle
import vpd_labels as VL  # noqa: E402
import vpd_model as VM  # noqa: E402

N = 2**24
REPLICAS = 16
CANDIDATES = 256  # per site: the subcomponents whose contribution changes most between the runs
ATTN, MLP = ("q_proj", "k_proj", "v_proj", "o_proj"), ("c_fc", "down_proj")


def site_name(l: int, kind: str) -> str:
    return f"h.{l}.{'mlp' if kind in MLP else 'attn'}.{kind}"


class Runner:
    def __init__(self):
        self.dev = VL.device()
        self.t = VL.Model(self.dev).t
        self.uv = VL.load_uv(self.dev, Path.home() / "mpd-data/oracle/vpd/uv.safetensors")

    def layer(self, i: int, x, patches: dict, clean: dict | None, reps: int, record: dict | None = None):
        """Layer i on residual x (reps copies of the batch); patches: {(i, kind): [idx tensor per copy]}
        set those subcomponents' contributions to the clean run's (clean: its site inputs)."""
        t = self.t
        H, D = t.n_head, t.hd
        Bx, L, _ = x.shape
        n = Bx // reps

        def site(kind, h):
            out = t.site(site_name(i, kind))(h)
            if record is not None:
                record[(i, kind)] = h
            for r, idx in enumerate(patches.get((i, kind), [])):
                if len(idx):
                    U, V = self.uv[site_name(i, kind)]
                    sl = slice(r * n, (r + 1) * n)
                    out[sl] = out[sl] + ((clean[(i, kind)] - h[sl]) @ V[:, idx]) @ U[idx]
            return out

        h = VM.rms(x, t.norms[2 * i], t.eps)
        q = site("q_proj", h).view(Bx, L, H, D).transpose(1, 2)
        k = site("k_proj", h).view(Bx, L, H, D).transpose(1, 2)
        v = site("v_proj", h).view(Bx, L, H, D).transpose(1, 2)
        q, k = t._rope(q, L), t._rope(k, L)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(Bx, L, -1)
        x = x + site("o_proj", y)
        hid = VM.gelu_tanh(site("c_fc", VM.rms(x, t.norms[2 * i + 1], t.eps)))
        return x + site("down_proj", hid)

    def forward(self, x, start: int, patches: dict, clean: dict | None, rows, cols, reps: int = 1,
                record: dict | None = None):
        """Log-probabilities at the targets, from layer `start` on (x: the residual entering it)."""
        t = self.t
        for i in range(start, t.n_layer):
            x = self.layer(i, x, patches, clean, reps, record)
        n = x.shape[0] // reps
        r = torch.cat([rows + c * n for c in range(reps)])
        return torch.log_softmax((VM.rms(x, t.ln_f, t.eps)[r, cols.repeat(reps)] @ t.wte.T).float(), -1)


@torch.no_grad()
def measure(run: Runner, beh: dict, blocks_measured: tuple[str, ...] = ("attn", "mlp")) -> dict:
    t, dev = run.t, run.dev
    prompts = [p for p in beh["prompts"] if p.get("counterfactual")]
    T = max(len(p["token_ids"]) for p in prompts)

    def batch(key):
        ids = torch.zeros(len(prompts), T, dtype=torch.long)
        for i, p in enumerate(prompts):
            x = (p if key == "clean" else p["counterfactual"])["token_ids"]
            ids[i, : len(x)] = torch.tensor(x)
        return ids.to(dev)

    clean_ids, cf_ids = batch("clean"), batch("cf")
    rows = torch.tensor([i for i, p in enumerate(prompts) for _ in p["target_positions"]], device=dev)
    cols = torch.tensor([c for p in prompts for c in p["target_positions"]], device=dev)
    clean = {}
    lp_clean = run.forward(t.wte[clean_ids], 0, {}, None, rows, cols, record=clean)
    cfrec = {}
    run.forward(t.wte[cf_ids], 0, {}, None, rows, cols, record=cfrec)
    p_clean = lp_clean.exp()
    entering = [t.wte[cf_ids]]  # the counterfactual residual entering each layer
    for i in range(t.n_layer - 1):
        entering.append(run.layer(i, entering[-1], {}, None, 1))

    def kl(lp, reps):
        per = (p_clean.repeat(reps, 1) * (lp_clean.repeat(reps, 1) - lp)).sum(-1) / math.log(2)
        return per.view(reps, -1).mean(1)

    base = kl(run.forward(entering[0], 0, {}, None, rows, cols), 1).item()
    recovery = {}
    for l in range(t.n_layer):
        for kind in (ATTN if "attn" in blocks_measured else ()) + (MLP if "mlp" in blocks_measured else ()):
            U, V = run.uv[site_name(l, kind)]
            change = ((clean[(l, kind)] - cfrec[(l, kind)]) @ V).abs().sum((0, 1)) * U.norm(dim=1)
            candidates = torch.argsort(change, descending=True)[:CANDIDATES].tolist()
            out = [0.0] * U.shape[0]
            for s in range(0, len(candidates), REPLICAS):
                sets = [torch.tensor([i], device=dev) for i in candidates[s : s + REPLICAS]]
                x = entering[l].repeat(len(sets), 1, 1)
                gains = (base - kl(run.forward(x, l, {(l, kind): sets}, clean, rows, cols, len(sets)), len(sets))).tolist()
                for i, g in zip(candidates[s : s + REPLICAS], gains):
                    out[i] = g
            recovery[f"{l}.{kind}"] = out
    d, m = t.wte.shape[1], run.uv[site_name(0, "c_fc")][0].shape[1]
    price = {k: (d + (m if k in MLP else d)) * 0.5 * math.log2(N) / N for k in ATTN + MLP}
    blocks = {}
    for l in range(t.n_layer):
        for name, kinds in [(b, ATTN if b == "attn" else MLP) for b in blocks_measured]:
            ranked = sorted(((recovery[f"{l}.{k}"][i], k, i) for k in kinds for i in range(len(recovery[f"{l}.{k}"]))),
                            reverse=True)
            curve, k = [], 1
            while k <= len(ranked):
                top = ranked[:k]
                sets = {(l, kind): [torch.tensor([i for _, kk, i in top if kk == kind], device=dev, dtype=torch.long)]
                        for kind in kinds}
                left = kl(run.forward(entering[l], l, sets, clean, rows, cols), 1).item()
                cost = sum(price[kk] for _, kk, _ in top)
                curve.append({"k": k, "kl_bits": left, "opaque_bits_per_token": cost, "total": left + cost})
                k *= 2
            best = min(curve, key=lambda c: c["total"])
            chosen = ranked[: best["k"]] if best["total"] < base else []
            blocks[f"{l}.{name}"] = {"curve": curve, "best": best,
                                     "chosen": {kk: sorted(i for _, k2, i in chosen if k2 == kk) for kk in kinds},
                                     "recovered_alone": sum(r for r, _, _ in chosen)}
    return {"behavior": beh["id"], "base_bits": base, "price": price, "recovery": recovery, "blocks": blocks}


def library_parts() -> dict[tuple[int, str], list[dict[str, list[int]]]]:
    """Decomp's start (arm grouped_own): per (layer, "attn" | "mlp") its parts in file order, each as
    {site kind: VPD subcomponent indices}."""
    start = Path.home() / "mpd-data/decomp/start.components.json"
    arm = next(r for r in json.loads(start.read_text()) if r["arm"] == "grouped_own")
    kinds = ATTN + MLP
    parts: dict = {}
    for c in arm["components"]:
        site = c["read"]["own"][0] if "own" in c["read"] else c["read"]["direction"]["site"]
        part: dict[str, list[int]] = {}
        for s_, k in c["slices"]:
            part.setdefault(kinds[s_ % 6], []).append(k)
        parts.setdefault((site // 6, "attn" if site % 6 < 4 else "mlp"), []).append(part)
    return parts


@torch.no_grad()
def measure_library(run: Runner, beh: dict, parts: dict) -> dict:
    """Each library part's recovery (all its slices patched together), and per block the top-k parts'
    joint patch curve with their opaque price."""
    t, dev = run.t, run.dev
    prompts = [p for p in beh["prompts"] if p.get("counterfactual")]
    T = max(len(p["token_ids"]) for p in prompts)

    def batch(key):
        ids = torch.zeros(len(prompts), T, dtype=torch.long)
        for i, p in enumerate(prompts):
            x = (p if key == "clean" else p["counterfactual"])["token_ids"]
            ids[i, : len(x)] = torch.tensor(x)
        return ids.to(dev)

    clean_ids, cf_ids = batch("clean"), batch("cf")
    rows = torch.tensor([i for i, p in enumerate(prompts) for _ in p["target_positions"]], device=dev)
    cols = torch.tensor([c for p in prompts for c in p["target_positions"]], device=dev)
    clean = {}
    lp_clean = run.forward(t.wte[clean_ids], 0, {}, None, rows, cols, record=clean)
    p_clean = lp_clean.exp()
    entering = [t.wte[cf_ids]]
    for i in range(t.n_layer - 1):
        entering.append(run.layer(i, entering[-1], {}, None, 1))

    def kl(lp, reps):
        per = (p_clean.repeat(reps, 1) * (lp_clean.repeat(reps, 1) - lp)).sum(-1) / math.log(2)
        return per.view(reps, -1).mean(1)

    def patches(l, chosen):  # chosen: list (one per replica) of parts -> {(l, kind): [idx per replica]}
        kinds = ATTN + MLP
        return {(l, k): [torch.tensor(sorted({i for part in group for i in part.get(k, [])}), device=dev, dtype=torch.long)
                         for group in chosen] for k in kinds}

    base = kl(run.forward(entering[0], 0, {}, None, rows, cols), 1).item()
    d, m = t.wte.shape[1], run.uv[site_name(0, "c_fc")][0].shape[1]
    price = {k: (d + (m if k in MLP else d)) * 0.5 * math.log2(N) / N for k in ATTN + MLP}
    blocks = {}
    for (l, name), block in sorted(parts.items()):
        recovery = []
        for s in range(0, len(block), REPLICAS):
            group = [[part] for part in block[s : s + REPLICAS]]
            x = entering[l].repeat(len(group), 1, 1)
            recovery += (base - kl(run.forward(x, l, patches(l, group), clean, rows, cols, len(group)), len(group))).tolist()
        order = sorted(range(len(block)), key=lambda i: -recovery[i])
        curve, k = [], 1
        while k <= len(block):
            top = [block[i] for i in order[:k]]
            left = kl(run.forward(entering[l], l, patches(l, [top]), clean, rows, cols, 1), 1).item()
            cost = sum(price[kk] * len(set(v)) for part in top for kk, v in part.items())
            curve.append({"k": k, "kl_bits": left, "opaque_bits_per_token": cost, "total": left + cost})
            k *= 2
        best = min(curve, key=lambda c: c["total"])
        blocks[f"{l}.{name}"] = {"recovery": recovery, "order": order, "curve": curve, "best": best,
                                 "chosen": sorted(order[: best["k"]]) if best["total"] < base else []}
    return {"behavior": beh["id"], "base_bits": base, "blocks": blocks}


def library_program(table: dict, beh: dict) -> str:
    """One node per block with chosen library parts (they read and write like a head or neurons)."""
    nodes = sorted(((int(k.split(".")[0]), k.split(".")[1], b) for k, b in table["blocks"].items() if b["chosen"]),
                   key=lambda n: (n[0], n[1] == "mlp"))
    pos = lambda n: 2 * n[0] + (n[1] == "mlp")  # noqa: E731
    name = lambda n: f"lib_{n[1]}{n[0]}"  # noqa: E731
    listed = "; ".join(f"layer {l} {nm}: {len(b['chosen'])} parts leaving {b['best']['kl_bits']:.2f} bits" for l, nm, b in nodes)
    doc = (f"Behavior {beh['id']} (vpd4l, our library: decomp's exact all-on start, arm grouped_own; M's top token "
           f"is right on {100 * beh['model_accuracy']:.0f}% of the targets): {beh['description']}\n\n"
           f"Nodes: per layer's attention and MLP, the library parts whose clean contribution (all their VPD slices), "
           f"patched into the counterfactual run, recovers the most of the {table['base_bits']:.2f} bits per target, as "
           f"many as pay for their opaque price: {listed}. The program lets every write among them reach every later read.")
    head, tail = doc.split("\n\n")
    out = ['"""' + "\n".join(textwrap.wrap(head, 100)) + "\n\n" + "\n".join(textwrap.wrap(tail, 100)) + '\n"""',
           "from mech import node, edges, PD, embed, logits", ""]
    for n in nodes:
        body = ", ".join(map(str, n[2]["chosen"]))
        out.append(f"{name(n)} = node(PD.lib[{n[0]}].{n[1]}[")
        out += [f"    {row}" for row in textwrap.wrap(body, 96)]
        out.append("])")
    wires = [f"embed >> {name(n)}" for n in nodes]
    wires += [f"{name(a)} >> {name(b)}" for a in nodes for b in nodes if pos(a) < pos(b)]
    wires += [f"{name(n)} >> logits" for n in nodes]
    out += ["", "edges("] + [f"    {w}," for w in wires] + [")", ""]
    return "\n".join(out)


def program(table: dict, beh: dict) -> str:
    """One node per block with chosen subcomponents; every causal edge among them."""
    nodes = []
    for key, b in table["blocks"].items():
        l, name = int(key.split(".")[0]), key.split(".")[1]
        pieces = {k: v for k, v in b["chosen"].items() if v}
        if pieces:
            nodes.append((l, name, pieces, b))
    nodes.sort(key=lambda n: (n[0], n[1] == "mlp"))
    reads = lambda ps: any(k in ps for k in ("q_proj", "k_proj", "v_proj", "c_fc"))  # noqa: E731
    writes = lambda ps: any(k in ps for k in ("o_proj", "down_proj"))  # noqa: E731
    pos = lambda n: 2 * n[0] + (n[1] == "mlp")  # noqa: E731
    name = lambda n: f"{n[1]}{n[0]}"  # noqa: E731
    listed = "; ".join(f"layer {l} {nm}: {sum(map(len, ps.values()))} subcomponents ({', '.join(f'{len(v)} {k}' for k, v in ps.items())}), "
                       f"leaving {b['best']['kl_bits']:.2f} bits when patched together" for l, nm, ps, b in nodes)
    doc = (f"Behavior {beh['id']} (vpd4l, VPD's subcomponents; M's top token is right on {100 * beh['model_accuracy']:.0f}% of "
           f"the targets): {beh['description']}\n\n"
           f"Nodes: per layer's attention and MLP, the VPD subcomponents whose clean contribution, patched into the "
           f"counterfactual run, recovers the most of the {table['base_bits']:.2f} bits per target between the clean and "
           f"counterfactual answers, as many as pay for their opaque price (1/2 log2 N bits per weight, N = 2^24): "
           f"{listed}. The program lets every write among them reach every later read.")
    head, tail = doc.split("\n\n")
    out = ['"""' + "\n".join(textwrap.wrap(head, 100)) + "\n\n" + "\n".join(textwrap.wrap(tail, 100)) + '\n"""',
           "from mech import node, edges, PD, embed, logits", ""]
    for n in nodes:
        l, _, ps, _ = n
        body = ", ".join(f"PD.vpd[{l}].{k}[{', '.join(map(str, v))}]" for k, v in ps.items())
        out.append(f"{name(n)} = node(")
        out += [f"    {row}" for row in textwrap.wrap(body, 96, break_long_words=False)]
        out.append(")")
    wires = [f"embed >> {name(n)}" for n in nodes if reads(n[2])]
    for a in nodes:
        for b in nodes:
            if pos(a) < pos(b) and writes(a[2]) and reads(b[2]):
                wires.append(f"{name(a)} >> {name(b)}")
    wires += [f"{name(n)} >> logits" for n in nodes if writes(n[2])]
    out += ["", "edges("] + [f"    {w}," for w in wires] + [")", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("behaviors", nargs="+", type=Path)
    ap.add_argument("--programs", type=Path)
    ap.add_argument("--library", action="store_true", help="measure decomp's library parts instead")
    ap.add_argument("--blocks", default="attn,mlp", help="which blocks' subcomponents to measure (attn, mlp)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    run = Runner()
    parts = library_parts() if a.library else None
    for path in a.behaviors:
        beh = json.loads(path.read_text())
        if (a.out_dir / f"{'libpatch' if a.library else 'vpdpatch'}_{beh['id']}.json").exists():
            continue  # measured already (a rerun after a stop goes on where it left off)
        if a.library:
            table = measure_library(run, beh, parts)
            (a.out_dir / f"libpatch_{beh['id']}.json").write_text(json.dumps(table))
            print(beh["id"], "base", round(table["base_bits"], 3),
                  {k: (b["best"]["k"], round(b["best"]["total"], 2)) for k, b in table["blocks"].items()}, flush=True)
            if a.programs:
                (a.programs / f"vpd4l_{beh['id'].replace('.', '_')}_lib.py").write_text(library_program(table, beh))
            continue
        table = measure(run, beh, tuple(a.blocks.split(",")))
        (a.out_dir / f"vpdpatch_{beh['id']}.json").write_text(json.dumps(table))
        print(beh["id"], "base", round(table["base_bits"], 3),
              {k: (b["best"]["k"], round(b["best"]["total"], 2)) for k, b in table["blocks"].items()}, flush=True)
        if a.programs:
            (a.programs / f"vpd4l_{beh['id'].replace('.', '_')}_vpd.py").write_text(program(table, beh))


if __name__ == "__main__":
    main()
