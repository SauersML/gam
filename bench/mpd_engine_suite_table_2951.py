"""The evaluation table of #2951's suite: every model's rows as they stand, gathered from the
mpd_engine_suite_2951 reports (~/mpd-data/engine/<model>/report.json: native bits, rounded
baselines, the n-ladder with explanation size, rollout fidelity and top components) and, for runs
with no finished rung, the status their logs end with (~/mpd-data/engine/runs/<model>.log), plus
VPD's matched rollout fidelity when bench/mpd_vpd_rollout_2951.py has written it.

usage: mpd_engine_suite_table_2951.py   (writes ~/mpd-data/engine/suite_table.json)
"""
import glob
import json
import os

E = os.path.expanduser("~/mpd-data/engine")
models = {}
for path in sorted(glob.glob(f"{E}/*/report.json")):
    r = json.load(open(path))
    name = os.path.basename(os.path.dirname(path))
    rungs = []
    for x in r.get("ladder", []):
        rungs.append({k: x.get(k) for k in (
            "observations", "program_bits", "structure_bits", "precision_bits", "explanation_bits", "data_bits",
            "total_bits", "best_rounded_total_bits", "best_rounded_b", "max_kl", "mean_kl", "max_tv",
            "argmax_agreement", "stop", "seconds", "active_per_input", "instances", "explanation_bits_per_input",
            "estimated_rows", "certified", "reals", "rollout")} | {"top_components": [
                {k: c.get(k) for k in ("name", "reads", "writes", "applied_by", "bits", "native")} for c in x.get("top_components", [])[:3]]})
    models[name] = {
        "model": r.get("model"), "rows": r.get("rows"), "native_bits": r.get("native", {}).get("program_bits"),
        "native": r.get("native"), "export_check": r.get("export_logits_max_centred_difference"),
        "reference_widest_band": r.get("reference_widest_band"), "seconds": r.get("seconds"),
        "rounded_at_one_observation": [{k: b.get(k) for k in ("b", "program_bits", "data_bits", "max_kl", "mean_kl", "argmax_agreement", "rollout")} for b in r.get("rounded_at_one_observation", [])],
        "ladder": rungs,
    }
# Runs with no completed rung yet, with what their logs say.
for log in sorted(glob.glob(f"{E}/runs/*.log")):
    name = os.path.basename(log)[:-4]
    if name == "table_updates" or (name in models and models[name].get("ladder")):
        continue
    text = open(log).read()
    status = "running (no rung completed)"
    for line in text.splitlines():
        if line.startswith("Error") or "killed" in line or line.startswith("STOPPED"):
            status = line[:300]
    head = next((l for l in text.splitlines() if "imported in" in l), "")
    models.setdefault(name, {})["status"] = status
    models[name]["log_head"] = head[:300]


def masked_rows(directory, export):
    """The masked-pieces path's rows (mpd_pieces_masked_2951's points.json), each full eval with its
    rollout fidelity: per eval row the summed per-token KL over the continuation the model sampled
    (positions PREFIX-1 .. PREFIX+HORIZON-2, export rollout.json) and greedy agreement there."""
    import numpy as np
    points = json.load(open(os.path.join(directory, "points.json")))["points"]
    rollout = json.load(open(os.path.join(export, "rollout.json")))
    record = json.load(open(os.path.join(export, "export.json")))
    width = record["files"]["tokens"]["shape"][1] - 1
    lo, hi = rollout["prefix"] - 1, rollout["prefix"] + rollout["horizon"] - 1
    for point in points:
        npass = point.get("pass")
        kl_path = os.path.join(directory, f"points.pass{npass}.kl.npy")
        agree_path = os.path.join(directory, f"points.pass{npass}.agree.npy")
        if npass is None or not os.path.exists(kl_path):
            continue
        kl = np.load(kl_path)
        context = kl.size // rollout["eval"]
        if context != width and context < hi:
            continue
        kl = kl.reshape(rollout["eval"], context)[:, lo:hi]
        agree = np.load(agree_path).reshape(rollout["eval"], context)[:, lo:hi]
        point["rollout"] = {"rollouts": rollout["eval"], "horizon": rollout["horizon"],
                            "sequence_kl_mean": float(kl.sum(1).mean()), "sequence_kl_max": float(kl.sum(1).max()),
                            "greedy_agreement": float(agree.mean())}
    return points


for name, export in (("pythia70m_masked", f"{E}/pythia70m_rollout"), ("vpd4l_masked", f"{E}/vpd4l_rollout")):
    if os.path.exists(f"{E}/{name}/points.json"):
        models[name] = {"path": "masked pieces (mpd_pieces_masked_2951)", "points": masked_rows(f"{E}/{name}", export)}
vpd = f"{E}/vpd4l_1x16/vpd_rollout.json"
out = {"models": models, "vpd_matched_rollouts": json.load(open(vpd)) if os.path.exists(vpd) else None}
json.dump(out, open(f"{E}/suite_table.json", "w"), indent=1)
print(len(models), "models;", ", ".join(f"{k}:{len(v.get('ladder', []))}" for k, v in models.items()))
