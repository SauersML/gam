"""E4 replication (#2951): the VPD paper's emoticon edit vs LoRA, by the paper's own protocol
(goodfire-ai/param-decomp, branch paper/editing-section: spd/editing/run_pareto_export.py, which wrote
figures/editing/pareto_data.json), then our pre-declared unrelated behaviours on top.

Protocol, as in the paper's code:
  examples   harvest-style windows: each firing of h.2.mlp.down_proj:2359 (CI > 0) with 20 tokens either side
             (clipped at the row), every firing inside the window flagged; the model sees ONLY the window
             (truncated context). Shuffled with random.seed(42); eval = first 50, train = next 947 (paper: 947).
  p_fire     mean P('o') at eval fire positions with a next token
  surr_kl    mean per-token KL(edited || target) over all NON-fire positions of the eval windows
  global_kl  mean per-token KL(edited || target) over every position of 40 full rows (paper: one train batch)
  VPD edit   U_c -> -alpha * u_o / |u_o| (bf16), i.e. W' = W + (new_u - U_c) V_c^T, alpha in {1.5,...,6}
  LoRA       rank 1 on h.2.mlp.down_proj (A ~ N(0, 0.01^2), B = 0), AdamW lr 1e-3, 300 steps, batch min(256, n)
             windows; loss = CE('o' at fire positions, 'o' written into the next token) + lambda * mean KL(edited
             || target) over the other unpadded positions; lambda in {0.1, 1, 10, 100}; n = 947 and n = 10.
Firing threshold: in our fp32 scans, 57% of CI > 0 firings have CI < 0.05 and sit mostly in non-emoticon
contexts (':' before a newline); the CI >= 0.7 firings are the emoticons the paper's examples show (its 30
heatmap windows fire on ' :' / ' ;' before 'D', 'P', '-)', '/'). E4_CI_MIN sets the firing threshold
(0 = CI > 0 as stated; 0.5 reproduces the paper's example mix); output e4_replication[_ciX].json.
Differences we cannot remove: windows come from Pile VAL rows (the paper harvested its training stream), our
scan of 28,672 rows gives 1337 firings, and batches of 256 are accumulated over micro-batches of 8 (exact).
Declared unrelated behaviours (fixed before any measurement, same as e4_edit_sideeffects.py): next-token CE on
'.', ' the', ' of', ',', digit tokens over 32 val rows at offset 40000 (excluding ':', ';', '=' contexts and
'o' targets) and induction on 32 repeated random sequences; damage = edited - unedited CE.
Writes e4_replication.json (paper numbers alongside).
usage: MPD_MEM_GIB=4 venv python mpd_vpd_edit_replication_2951.py
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "vpd_2951"))
from vpd_model import VPD_PTH, load_target, val_tokens  # noqa: E402

DEV, SITE, COMP, O_TOK, SIDE = "mps", "h.2.mlp.down_proj", 2359, 80, 20
VD = Path.home() / "mpd-data/vpd"
CI_MIN = float(__import__("os").environ.get("E4_CI_MIN", "0"))  # firing = CI > CI_MIN (the scans store CI x 1e6)
OUT = Path.home() / ("mpd-data/frontier/e4_replication.json" if CI_MIN == 0 else f"mpd-data/frontier/e4_replication_ci{CI_MIN:g}.json")
PAPER = json.load(open(VD / "paper_editing/paper__editing-section/figures/editing/pareto_data.json"))
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

target = load_target(DEV)
site = target.site(SITE)
raw = torch.load(str(VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
key = "_components." + SITE.replace(".", "-")
U = raw[key + ".U"].float().to(DEV)[COMP]  # [d_out]
V = raw[key + ".V"].float().to(DEV)[:, COMP]  # [d_in]
del raw
W0 = site.W.clone()
u_o = target.wte[O_TOK] / target.wte[O_TOK].norm()  # tied unembedding row

# ---------------------------------------------------------------- harvest-style windows
fires = {}
for scan in ("scan_0_4096.pt", "scan_edits_4096_28672.pt"):
    s = torch.load(VD / scan)
    for r, p, ci in s["fires"][(SITE, COMP)].tolist():
        if ci / 1e6 <= CI_MIN:
            continue
        fires.setdefault(r, set()).add(p)
rows_needed = sorted(fires)
_rz = np.load(Path.home() / "mpd-data/frontier/e4_rows.npz")  # written by mpd_vpd_edit_rows_2951.py (no parquet in this process)
row_ids = {int(r): torch.tensor(t, dtype=torch.long) for r, t in zip(_rz["rows"], _rz["tokens"])}
assert all(r in row_ids for r in rows_needed)
examples = []
for r in rows_needed:
    for p in sorted(fires[r]):
        a, b = max(0, p - SIDE), min(512, p + SIDE + 1)
        toks = row_ids[r][a:b]
        f = [q - a for q in sorted(fires[r]) if a <= q < b]
        if any(q + 1 < len(toks) for q in f):
            examples.append((toks, f))
random.seed(42)
random.shuffle(examples)
ev, train = examples[:50], examples[50:50 + 947]
log(f"{len(examples)} windows; eval {len(ev)}, train {len(train)}")
glob = val_tokens(40, offset=2048)

# ---------------------------------------------------------------- declared unrelated behaviours
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(str(VD / "t-9d2b8f02/tokenizer.json"))
DECL_IDS = torch.tensor(_rz["decl"], dtype=torch.long)  # val rows 40000-40031 (from e4_rows.npz)
_g = torch.Generator().manual_seed(0)
_rand = torch.randint(1000, 30000, (32, 64), generator=_g)
IND_IDS = torch.cat([_rand, _rand], 1)
_s = lambda t: (tok.id_to_token(int(t)) or "").replace("Ġ", "").strip()
_cur_bad = torch.tensor([[any(ch in _s(t) for ch in ":;=") for t in row] for row in DECL_IDS.tolist()])
_nxt = DECL_IDS[:, 1:]
_ok = ~_cur_bad[:, :-1] & (_nxt != O_TOK)
DECL_MASKS = {"period": _ok & (_nxt == tok.token_to_id(".")), "the": _ok & (_nxt == tok.token_to_id("Ġthe")),
              "of": _ok & (_nxt == tok.token_to_id("Ġof")), "comma": _ok & (_nxt == tok.token_to_id(",")),
              "digits": _ok & torch.tensor([[_s(t).isdigit() for t in row] for row in _nxt.tolist()])}


@torch.no_grad()
def declared_ce():
    out = {k: 0.0 for k in DECL_MASKS}
    for i in range(0, 32, 2):  # 2 rows at a time: vocab-sized logits are the memory peak
        lp = F.log_softmax(target(DECL_IDS[i:i + 2].to(DEV)), -1)[:, :-1]
        ce = -lp.gather(-1, DECL_IDS[i:i + 2, 1:].to(DEV)[..., None])[..., 0].cpu()
        for k, m in DECL_MASKS.items():
            out[k] += ce[m[i:i + 2]].sum().item() / max(m.sum().item(), 1)
        del lp
    ind = 0.0
    for i in range(0, 32, 8):
        lp = F.log_softmax(target(IND_IDS[i:i + 8].to(DEV)), -1)
        ind += (-lp[:, 64:-1].gather(-1, IND_IDS[i:i + 8, 65:].to(DEV)[..., None])[..., 0]).sum().item()
        del lp
    out["induction"] = ind / (32 * 63)
    torch.mps.empty_cache()
    return out


def kl_ed_base(lp_e, lp_b):
    """The paper's per-token KL(edited || base)."""
    return (lp_e.exp() * (lp_e - lp_b)).sum(-1)


@torch.no_grad()
def evaluate(W):
    """p_fire, surr_kl, global_kl and declared CE under weights W (the paper's eval_edit)."""
    p_fire, surr, glo = [], [], []
    for toks, f in ev:
        b = toks[None].to(DEV)
        site.W = W0
        lb = F.log_softmax(target(b)[0], -1)
        site.W = W
        le = F.log_softmax(target(b)[0], -1)
        kl = kl_ed_base(le, lb)
        for q in f:
            if q + 1 < len(toks):
                p_fire.append(le[q, O_TOK].exp().item())
        keep = torch.ones(len(toks), dtype=torch.bool)
        keep[f] = False
        surr += kl[keep.to(DEV)].tolist()
    for i in range(0, 40, 2):
        b = glob[i:i + 2].to(DEV)
        site.W = W0
        lb = F.log_softmax(target(b), -1)
        site.W = W
        le = F.log_softmax(target(b), -1)
        glo.append(kl_ed_base(le, lb).mean().item())
        del lb, le
    d = declared_ce()
    site.W = W0
    return {"p_fire": float(np.mean(p_fire)), "surr_kl": float(np.mean(surr)), "global_kl": float(np.mean(glo)), "declared_ce": d}


def train_lora(pool, lam, steps=300, batch=256, micro=8):
    """The paper's LoRATrainer (rank 1, AdamW lr 1e-3), batches of min(256, n) accumulated over micro-batches."""
    torch.manual_seed(0)
    A = (torch.randn(1, W0.shape[1]) * 0.01).to(DEV).requires_grad_(True)
    B = torch.zeros(W0.shape[0], 1, device=DEV, requires_grad=True)
    opt = torch.optim.AdamW([A, B], lr=1e-3)
    seqs = []
    for toks, f in pool:  # write 'o' after every fire position (make_train_seqs)
        t = toks.clone()
        fp = [q for q in f if q + 1 < len(t)]
        for q in fp:
            t[q + 1] = O_TOK
        seqs.append((t, fp))
    L = max(len(t) for t, _ in seqs)
    T = torch.zeros(len(seqs), L, dtype=torch.long)
    fire = torch.zeros(len(seqs), L, dtype=torch.bool)
    pad = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, (t, fp) in enumerate(seqs):
        T[i, :len(t)] = t
        fire[i, fp] = True
        pad[i, :len(t)] = True
    n = len(seqs)
    for step in range(steps):
        idx = torch.randint(n, (min(batch, n),))
        n_fire = int(fire[idx].sum())
        n_kl = int((pad[idx] & ~fire[idx]).sum())
        opt.zero_grad()
        for j in range(0, len(idx), micro):
            mi = idx[j:j + micro]
            tb, fb, pb = T[mi].to(DEV), fire[mi].to(DEV), pad[mi].to(DEV)
            with torch.no_grad():
                site.W = W0
                lb = F.log_softmax(target(tb), -1)
            site.W = W0 + B @ A
            lg = target(tb)
            fi = fb.nonzero()
            ce = F.cross_entropy(lg[fi[:, 0], fi[:, 1]], tb[fi[:, 0], fi[:, 1] + 1], reduction="sum") / max(n_fire, 1)
            le = F.log_softmax(lg, -1)
            klm = pb & ~fb
            kl = kl_ed_base(le, lb)[klm].sum() / max(n_kl, 1)
            (ce + lam * kl).backward()
            del lg, le, lb
        opt.step()
    site.W = W0
    return (B @ A).detach()


base_decl = declared_ce()
res = {"description": __doc__, "paper": PAPER, "n_windows": len(examples), "n_eval": len(ev), "n_train": len(train),
       "declared_baseline_ce": base_decl, "vpd": {}, "lora": {}, "lora_low": {}}
if OUT.exists():  # resume: keep every point already measured (runs get killed under machine-wide swapping)
    old = json.load(open(OUT))
    for k in ("vpd", "lora", "lora_low"):
        res[k].update(old.get(k, {}))


def add_damage(d):
    d["declared_damage"] = {k: v - base_decl[k] for k, v in d["declared_ce"].items()}
    d["declared_abs_damage_mean"] = float(np.mean([abs(x) for x in d["declared_damage"].values()]))
    return d


for a in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
    if str(a) in res["vpd"]:
        continue
    new_u = (-a * u_o).to(torch.bfloat16).float()
    r = add_damage(evaluate(W0 + torch.outer(new_u - U, V)))
    res["vpd"][str(a)] = r
    log(f"VPD alpha {a}: p_fire {r['p_fire']:.3f} (paper {PAPER['spd'][str(a)]['p_fire']:.3f}) surr {r['surr_kl']:.5f} "
        f"({PAPER['spd'][str(a)]['surr_kl']:.5f}) global {r['global_kl']:.5f} ({PAPER['spd'][str(a)]['global_kl']:.5f}) damage {r['declared_abs_damage_mean']:.4f}")
    json.dump(res, open(OUT, "w"), indent=1)
for key, pool in (("lora", train), ("lora_low", train[:10])):
    for lam in (0.1, 1.0, 10.0, 100.0):
        if str(lam) in res[key]:
            continue
        dW = train_lora(pool, lam)
        r = add_damage(evaluate(W0 + dW))
        res[key][str(lam)] = r
        pp = PAPER[key][str(lam)]
        log(f"{key} lambda {lam}: p_fire {r['p_fire']:.3f} (paper {pp['p_fire']:.3f}) surr {r['surr_kl']:.5f} ({pp['surr_kl']:.5f}) "
            f"global {r['global_kl']:.5f} ({pp['global_kl']:.5f}) damage {r['declared_abs_damage_mean']:.4f}")
        json.dump(res, open(OUT, "w"), indent=1)
log("done")
