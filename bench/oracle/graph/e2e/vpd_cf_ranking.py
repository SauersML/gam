"""Counterfactual ranking of VPD subcomponents for a behavior (#2951, night plan R2): how much each
subcomponent's write changes between a prompt x and its counterfactual x', which is exactly what a
counterfactual stand-in replaces when the subcomponent is left out of a program.

For subcomponent i at a site with input h (the site's normed input), its write at position t is
(h_t . v_i) u_i; the change is |(h_t(x) - h_t(x')) . v_i| ||u_i||, summed over the positions up to each
target and averaged over the behavior's prompts. Both runs are the model's own forward passes (vpd4l in
torch, bench/oracle/vpd_labels.py); no gradients. Output in removal-scan form for search.py --ranking:
{"sites": {site: {"cf_write_change": [per subcomponent]}}, "behavior": id}.

  vpd_cf_ranking.py BEHAVIOR.json OUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # bench/oracle
import vpd_labels as VL  # noqa: E402
import vpd_model as VM  # noqa: E402


@torch.no_grad()
def main():
    behavior = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    dev = VL.device()
    model = VL.Model(dev)
    uv = VL.load_uv(dev, Path.home() / "mpd-data/oracle/vpd/uv.safetensors")
    names = VM.site_names(model.t.n_layer)
    total = {n: torch.zeros(uv[n][0].shape[0], dtype=torch.float32, device=dev) for n in names}
    used = 0
    for p in behavior["prompts"]:
        cf = p.get("counterfactual")
        if not cf or len(cf["token_ids"]) != len(p["token_ids"]):
            continue
        last = max(p["target_positions"]) + 1
        records = []
        for ids in (p["token_ids"][:last], cf["token_ids"][:last]):
            x = model.t.wte[torch.tensor([ids], device=dev)]
            rec: dict = {}
            for i in range(model.t.n_layer):
                x, _ = model.layer(i, x, record=rec)
            records.append(rec)
        for n in names:
            U, V = uv[n]  # U [C, d_out], V [d_in, C]
            dh = (records[0][n] - records[1][n])[0]  # [T, d_in]
            change = (dh @ V).abs().sum(0) * U.norm(dim=1)  # [C]
            total[n] += change.float()
        used += 1
    result = {"behavior": behavior["id"], "prompts": used,
              "sites": {n: {"cf_write_change": (total[n] / max(used, 1)).tolist()} for n in names}}
    out.write_text(json.dumps(result))
    top = sorted(((v, n, i) for n in names for i, v in enumerate(result["sites"][n]["cf_write_change"])), reverse=True)[:8]
    print(f"{used} prompts; top: " + ", ".join(f"{n}[{i}] {v:.3g}" for v, n, i in top))


if __name__ == "__main__":
    main()
