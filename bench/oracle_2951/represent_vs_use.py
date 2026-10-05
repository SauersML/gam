"""Per head of vpd4l (#2951): two activation readouts (attention, from a repeated token, to the token
after its earlier occurrence; the head's direct logit for the copied token) against the measured
crossed share of the copy: x1 = 24 random tokens with A at position i followed by B, then A; x0 = the
same with B replaced; share = -gamma / input effect with the head's output map at zero.

usage: MPD_MEM_GIB=1 venv python represent_vs_use.py SESSION.json OUT.json   (a session on vpd4l, model "model")
"""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import oracle
s = oracle.Session(sys.argv[1])
rng = np.random.default_rng(7)
pairs, meta = [], []
for _ in range(32):
    toks = [int(t) for t in rng.integers(1000, 20000, size=24)]
    i = int(rng.integers(4, 14))
    a, b, b2 = toks[i], toks[i + 1], int(rng.integers(1000, 20000))
    x1 = toks + [a]
    x0 = list(x1); x0[i + 1] = b2
    pairs.append({"x0": x0, "x1": x1, "target": b})
    meta.append(i)
heads = [(l, h) for l in range(4) for h in range(6)]
out = {"heads": [], "pairs": len(pairs)}
base = s.request({"op": "crossed", "model": "model", "pairs": pairs, "a1": {"edits": []}})
for (l, h) in heads:
    r = s.request({"op": "crossed", "model": "model", "pairs": pairs, "a1": {"edits": [{"component": {"kind": "head", "layer": l, "head": h}, "alpha": 0.0}]}})
    att = []
    for p, i in zip(pairs, meta):
        a = s.request({"op": "attention", "model": "model", "tokens": p["x1"], "layer": l, "head": h, "top": 30})
        last = a["rows"][-1]
        att.append(dict((src, w) for src, w in last["sources"]).get(i + 1, 0.0))
    # direct logit readout of the head's write on the copied token at the last position
    dla = []
    for p in pairs[:12]:
        u = s.request({"op": "unembed", "model": "model", "site": {"kind": "head", "layer": l, "head": h}, "tokens": p["x1"], "position": -1, "top": 50277})
        scores = dict((t, v) for t, v in u["promoted"])
        dla.append(scores.get(p["target"], 0.0))
    e = r["input_effect_a0"]["mean"]
    out["heads"].append({"layer": l, "head": h, "effect": e, "gamma": r["gamma"]["mean"], "gamma_se": r["gamma"]["standard_error"],
                         "share": -r["gamma"]["mean"] / e, "share_se": r["gamma"]["standard_error"] / abs(e),
                         "attention_to_successor": float(np.mean(att)), "direct_logit": float(np.mean(dla))})
    print(l, h, round(-r["gamma"]["mean"] / e, 3), round(float(np.mean(att)), 3), round(float(np.mean(dla)), 3), flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
