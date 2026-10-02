"""Collect the VPD side of the pre-registered scoreboard into scoreboard.json (and print it)."""

import json
import os

import numpy as np

J = lambda p: json.load(open(p)) if os.path.exists(p) else None
paper = json.load(open("s-55ea3f9b/wandb-summary.json"))
jax = json.load(open("p-8383f5e5/wandb-summary.json"))
out = {"prereg": {
    "S2_L0_analogue": "per token: number of (site, rank-one factor) usages with nonzero gate, summed "
                      "over the 24 sites; for rule/program outputs each firing rule call = 1 usage",
    "S1": "bits = LatticeCode (Elias-omega, per-tensor p = b - round(log2 rms)) over every real the "
          "decoder needs (VPD: U, V at 24 sites + the CI transformer; delta excluded since it needs W); "
          "KL vs the fp32 target on val rows 1024..1055",
}}
t = J("table_4x128.json")
if t:
    B = t["batches"]
    mean = lambda k: float(np.mean([b[k] for b in B]))
    keys = [k for k in B[0] if isinstance(B[0][k], float)]
    out["standard_table_512x512"] = {k: mean(k) for k in keys}
    out["standard_table_512x512"]["attn_kl_layers_ci"] = np.mean([b["CIAttnPatternsReconLoss/layers"] for b in B], 0).tolist()
    out["standard_table_512x512"]["attn_kl_layers_stoch"] = np.mean([b["StochasticAttnPatternsReconLoss/layers"] for b in B], 0).tolist()
    out["alive"] = {k: t[k] for k in ("alive_total_meanci_gt_1e-6", "alive_total_fired_once", "n_tokens_usage")}
    a = t["alive_meanci_gt_1e-6"]
    out["alive"]["per_layer"] = [sum(v for n, v in a.items() if n.startswith(f"h.{l}.")) for l in range(4)]
r = J("repro_4x128.json")
if r:
    out["pgd20_shared_0.1"] = {"with_delta": [x["pgd_with_delta"]["20"] for x in r],
                               "no_delta": [x["pgd_no_delta"]["20"] for x in r]}
out["paper_run_summary"] = {k.split("/")[-1]: v for k, v in paper.items() if k.startswith("eval/ce_kl/kl_") or k == "eval/loss/PGDReconLoss"}
out["jax_reference_summary"] = {k.split("/")[-1]: v for k, v in jax.items() if k.startswith("eval/ce_kl/kl_") or k.startswith("eval/loss/PGDReconLoss")}
for name in ("ppgd", "pgd_ladder", "prune", "cases", "edit", "frontier", "pareto_a", "pareto_b", "interventions", "pertoken", "ladder_vpd"):
    v = J(name + ".json")
    if v is not None:
        out[name] = v
json.dump(out, open("scoreboard.json", "w"), indent=1)



# ------------------------------------------------------------------ compact table (numbers only)
rows = []
put = lambda k, v: rows.append((k, v))
st = out.get("standard_table_512x512", {})
if st:
    ct = st["ce_target"]
    put("target CE (512x512 val rows)", ct)
    for s in ("unmasked", "stoch_masked", "rounded_masked", "ci_masked", "zero_masked"):
        put(f"VPD CE {s}", ct + st[f"ce_difference_{s}"])
    for s in ("unmasked", "stoch_masked", "rounded_masked", "ci_masked"):
        put(f"VPD KL {s}", st[f"kl_{s}"])
    for s in ("unmasked", "stoch_masked", "rounded_masked", "ci_masked"):
        put(f"VPD top-1 agreement {s}", st[f"top1_agree_{s}"])
    put("VPD L0 (gates > 0, per token)", float(np.mean([b["l0_total"] for b in t["batches"]])))
    put("VPD alive components (mean CI > 1e-6)", out["alive"]["alive_total_meanci_gt_1e-6"])
if "pgd20_shared_0.1" in out:
    put("VPD PGD-20 KL (shared, 0.1, with delta)", float(np.mean(out["pgd20_shared_0.1"]["with_delta"])))
    put("VPD PGD-20 KL (shared, 0.1, no delta)", float(np.mean(out["pgd20_shared_0.1"]["no_delta"])))
if out.get("pgd_ladder"):
    for k, v in sorted(out["pgd_ladder"]["pgd"].items(), key=lambda kv: int(kv[0])):
        put(f"VPD PGD-{k} KL (shared, 0.1, with delta, 128 rows)", v)
if out.get("ppgd"):
    for k, v in out["ppgd"]["ppgd"].items():
        put(f"VPD PPGD-{k} KL (per-position, 16 rows)", v)
order = lambda r: (r["b"].startswith("dense"), -(32 if r["b"] == "fp32" else int(r["b"].removeprefix("dense"))))
fr = sorted(out.get("frontier") or [], key=order)
for r in fr:
    if r.get("l0_total") is not None:
        put(f"S1 VPD b={r['b']}: bits U,V | Gamma | total", (r["bits_uv"], r["bits_ci"], r["bits_total"]))
        put(f"S1 VPD b={r['b']}: KL ci | rounded | PGD-20 (no delta)", (r["kl_ci_masked"], r["kl_rounded_masked"], r["pgd20"]))
    else:
        put(f"S1 dense target b={r['b'][5:]}: bits | KL", (r["bits_total"], r["kl_unmasked"]))
pts = [p for name in ("pareto_a", "pareto_b") if out.get(name) for p in out[name]["points"]]
for p in sorted({p["tau"]: p for p in pts}.values(), key=lambda p: p["tau"]):
    put(f"S2 VPD tau={p['tau']}: L0 | KL ci | KL rounded | PGD-20", (p["l0_total"], p["kl_ci_masked"], p["kl_rounded_masked"], p["pgd20_shared_no_delta"]))
pr = out.get("prune")
if pr:
    for k, v in pr["static_neurons"].items():
        put(f"S2 pruned static neurons/MLP={k}: KL", v)
    for k, v in pr["dynamic_neurons"].items():
        put(f"S2 pruned dynamic top-k neurons/MLP={k}: KL", v)
    for k, v in pr["static_heads"].items():
        put(f"S2 pruned static heads/layer={k}: KL", v)
ed = out.get("edit")
if ed:
    for c, r in ed["comps"].items():
        if "baseline" not in r:
            continue
        put(f"S3 comp {c} baseline: p(o) | KL surround | KL global", tuple(r["baseline"][k] for k in ("p_o", "kl_surround", "kl_global")))
        for e in r["vpd_edit"]:
            put(f"S3 comp {c} VPD edit alpha={e['alpha']}", (e["p_o"], e["kl_surround"], e["kl_global"]))
        for e in r.get("lora", []):
            put(f"S3 comp {c} LoRA n={e.get('n')} r={e.get('rank')} lam={e.get('lambda')}", (e["p_o"], e["kl_surround"], e["kl_global"]))
cs = out.get("cases")
if cs:
    tg, ab = cs["target"], cs["ablations"]
    put("S4 L1 prev-token attention per head, target", tuple(tg["prev1"]))
    for comp in ("q:316", "k:329", "q:316+k:329", "q:308"):
        if comp in ab:
            put(f"S4 L1 prev-token attention per head, ablate {comp}", tuple(ab[comp]["prev1"]))
    ctl = cs.get("control_q") or {}
    if ctl:
        put("S4 L1 head-1 prev-token attn, 20 random q-comp ablations: min | max",
            (min(v["prev1"][1] for v in ctl.values()), max(v["prev1"][1] for v in ctl.values())))
iv = out.get("interventions")
if iv:
    for case in ("gender", "gender_prince"):
        c = iv[case]
        put(f"S4 '{c['prompt']}' target: p(her) | p(his)", (c["target"][" her"], c["target"][" his"]))
        for name, a in c["ablations"].items():
            put(f"S4 '{c['prompt']}' ablate {name}: p(her) | p(his)", (a[" her"], a[" his"]))
    c = iv["bracket"]
    put(f"S4 '{c['prompt']}' target: p(>)", c["target"][">"])
    for name, a in c["ablations"].items():
        put(f"S4 '{c['prompt']}' ablate {name}: p(>)", a[">"])
pt = out.get("pertoken")
if pt:
    bt = json.load(open("bits_table.json"))
    lib = {"fp32": 32 * bt["2"]["uv"]["n_reals"], "2": bt["2"]["uv"]["bits"]}
    for key, r in pt["variants"].items():
        put(f"PT VPD {key}: KL | top-1 | L0 | bits/token (set + coef)",
            (r["kl"], r["top1"], r["l0"], r["bits_per_token"], r["bits_set_per_token"], r["bits_coef_per_token"]))
        L = lib[r["uv"]]
        put(f"PT VPD {key}: library bits (U,V, no Gamma) | amortized bits/token at N = 1.6e4 | 1e6 | 1e9",
            (L, L / pt["n_tokens"] + r["bits_per_token"], L / 1e6 + r["bits_per_token"], L / 1e9 + r["bits_per_token"]))
    for r in fr:
        if r.get("l0_total") is None:
            L = r["bits_total"]
            put(f"PT dense target {r['b']}: KL | bits/token | library | amortized at N = 1.6e4 | 1e6 | 1e9",
                (r["kl_unmasked"], 0.0, L, L / pt["n_tokens"], L / 1e6, L / 1e9))
lv = out.get("ladder_vpd")
if lv:
    for k, v in lv["ladder_mean"].items():
        put(f"VPD input-space ladder (CEGAR adversary) step {k}: mean | worst KL", (v, lv["ladder_worst"][k]))
json.dump(rows, open("scoreboard_compact.json", "w"), indent=0)
fmt = lambda v: " | ".join(fmt(x) for x in v) if isinstance(v, (tuple, list)) else (f"{v:.4g}" if isinstance(v, float) else str(v))
for k, v in rows:
    print(f"{k:62s} {fmt(v)}")
