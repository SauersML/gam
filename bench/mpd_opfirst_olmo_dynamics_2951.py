"""#2951 operator-first probes across OLMo 2 1B training checkpoints: does structure emerge, sharpen or wash out?

Runs three cheap probes on each revision (one checkpoint loaded at a time, receipts kept under --work) and
aggregates them into one receipt:

* head sharing (bench/mpd_opfirst_rope_span_2951.py, weights only, exact operator algebra): per layer the
  participation ratio of the QK-operator Gram against the "every head its own orthogonal subspace" null
  (ratio < 1 = energy shared across heads) and the mean fraction of one head's operator energy captured by
  another head's span;
* sign-gated law share (bench/mpd_opfirst_mlp_oddeven_2951.py --mode relusplit, empirical on 512 fineweb-edu
  tokens): variance of the MLP output explained by the relu part P = W_d[relu(g) * u], and of the actual
  residual write N_ff(F) by N_ff(P) (OLMo 2 rescales the MLP output before the add);
* Pi-graph / E* split vs twin on MLP up reads (bench/mpd_opfirst_gelu_modules_2951.py --mode pi --read up,
  weights only): best Fiedler-proposal E* over the random-subset mean and over the same proposer on a twin
  with redrawn unit directions. NOT exact for SwiGLU (the exact-GELU module theorem does not cover a
  two-read bilinear-gated unit); it measures read-row geometry only.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpd_opfirst_decoder_2951 import compact_json  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REVISIONS = ("stage1-step0-tokens0B,stage1-step300-tokens1B,stage1-step10000-tokens21B,stage1-step100000-tokens210B,"
             "stage1-step1000000-tokens2098B,stage1-step1907359-tokens4001B,main")


def run(script, *args):
    subprocess.run([sys.executable, os.path.join(HERE, script), *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--revisions", default=REVISIONS)
    ap.add_argument("--pi-layers", default="1,14")
    ap.add_argument("--pi-fracs", default="1/64,1/16,1/4")
    ap.add_argument("--work", default=os.path.expanduser("~/mpd-data/olmo/dynamics"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    t0 = time.time()
    out = {"model": args.model,
           "labels": {"head_sharing": "exact operator algebra on weights (q/k-norm normalisers excluded)",
                      "relu_share": "empirical, 512 fineweb-edu tokens, position 0 excluded, float64 analysis of a "
                                    "float32 forward",
                      "pi_up": "weights only; NOT an exact module statement for SwiGLU (read-row geometry of W_up)"},
           "revisions": []}
    for rev in args.revisions.split(","):
        paths = {k: os.path.join(args.work, f"{k}_{rev}.json") for k in ("rope", "relu", "pi")}
        if not os.path.exists(paths["rope"]):
            run("mpd_opfirst_rope_span_2951.py", "--model", args.model, "--revision", rev, "--out", paths["rope"])
        if not os.path.exists(paths["relu"]):
            run("mpd_opfirst_mlp_oddeven_2951.py", "--model", args.model, "--revision", rev, "--mode", "relusplit",
                "--out", paths["relu"])
        if not os.path.exists(paths["pi"]):
            run("mpd_opfirst_gelu_modules_2951.py", "--model", args.model, "--revision", rev, "--mode", "pi",
                "--read", "up", "--layers", args.pi_layers, "--fracs", args.pi_fracs, "--random-subsets", "10",
                "--out", paths["pi"])
        rope, relu, pi = (json.load(open(paths[k])) for k in ("rope", "relu", "pi"))
        if not relu["text_source"].startswith("HuggingFaceFW/fineweb-edu"):  # every revision reads the same tokens
            os.remove(paths["relu"])
            raise SystemExit(f"{rev}: relusplit fell back to {relu['text_source']!r}; rerun when fineweb is reachable")
        layers = rope["layers"]
        relu_layers = [relu["layers"][str(i)] for i in range(len(layers))]
        rec = {
            "revision": rev,
            "head_pr_ratio": [r["actual"]["participation_ratio"] / r["null_heads_independent"]["participation_ratio"]
                              for r in layers],
            "head_captured_diff": [r["head_pairs"]["diff_k"]["captured"] for r in layers],
            "qk_span_rank": [r["rank"]["numerical_rank"] for r in layers],
            "relu_fve_P": [r["fve_P_centered"] for r in relu_layers],
            "relu_fve_postnorm_write": [r["post_norm_write"]["fve_NP_centered"] for r in relu_layers],
            "relu_active_frac": [r["active_frac_tok"]["median"] for r in relu_layers],
            "pi_up": [{"layer": r["layer"], "components_3e-2": r["pi_components_band"]["0.03"]["components"],
                       "by_k": [{"k": b["k"], "ratio_vs_random": b["ratio_vs_random"],
                                 "ratio_vs_twin": b["ratio_vs_twin"]} for b in r["by_k"]]} for r in pi["layers"]],
            "text_source": relu["text_source"],
        }
        out["revisions"].append(rec)
        print(rev, "PRratio", [round(v, 2) for v in rec["head_pr_ratio"]], "fveP",
              [round(v, 2) for v in rec["relu_fve_P"]], flush=True)
    out["runtime_s"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(compact_json(out))


if __name__ == "__main__":
    main()
