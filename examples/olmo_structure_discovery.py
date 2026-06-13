"""Anytime-valid structure discovery: is exp/noexp a REAL structure in OLMo?

SCIENCE QUESTION: Does the exp/noexp distinction correspond to a real
separable structure in L25 activations, or is it noise? We use gam's
e-process (universal inference, anytime-valid) to answer this with
valid sequential inference across checkpoints.

METHOD:
1. For each checkpoint, split into train/eval folds.
2. Fit a null model (no exp/noexp distinction) on train, evaluate on eval.
3. Fit an alternative model (exp side as a predictor) on train, evaluate on eval.
4. Feed log-likelihoods into an AtomBirthGate → e-process.
5. Run across all checkpoints cumulatively (optional-stopping safe).

Additionally: use e_bh_dictionary_certificate to test which specific
entity-kinds (self, human, ai, rock, tool, ...) show real structure.

Outputs:
  /projects/standard/hsiehph/sauer354/olmo_data/plots/structure_discovery.png
  /projects/standard/hsiehph/sauer354/olmo_data/plots/structure_discovery.json
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

DATA_DIR = Path("/projects/standard/hsiehph/sauer354/olmo_data")
OUT_DIR = Path("/projects/standard/hsiehph/sauer354/olmo_data/plots")
CHECKPOINTS = ["base", "step_2300", "stage1-step0", "stage3-step11921"]
ALPHA = 0.05
PCA_DIM = 8
RANDOM_STATE = 42


def pca_whiten(X: np.ndarray, n: int) -> np.ndarray:
    mu = X.mean(0)
    Xc = (X - mu).astype(np.float64)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:n].T) / (S[:n] + 1e-12)


def gaussian_log_lik(residuals: np.ndarray) -> float:
    """Log-likelihood under MLE Gaussian (closed form)."""
    n = len(residuals)
    sigma2 = float(np.var(residuals))
    if sigma2 <= 0:
        return float("nan")
    return float(-0.5 * n * (np.log(2 * np.pi * sigma2) + 1))


def fit_models_and_log_liks(
    Z_train: np.ndarray, Z_eval: np.ndarray,
    sides_train: np.ndarray, sides_eval: np.ndarray,
    label: str,
) -> tuple[float, float]:
    """
    Returns (log_lik_alt_on_eval, log_lik_null_sup_on_eval).

    Null: response (PC1 of eval) predicted from PCs 2..d only (no side info).
    Alt: same + is_exp indicator.

    We fit OLS on train, evaluate on eval.
    """
    y_train = Z_train[:, 0]
    y_eval = Z_eval[:, 0]
    X_null_train = Z_train[:, 1:]
    X_null_eval = Z_eval[:, 1:]
    is_exp_train = (sides_train == "exp").astype(float).reshape(-1, 1)
    is_exp_eval = (sides_eval == "exp").astype(float).reshape(-1, 1)
    X_alt_train = np.hstack([X_null_train, is_exp_train])
    X_alt_eval = np.hstack([X_null_eval, is_exp_eval])

    # OLS fits (ridge for stability)
    ridge = 1e-4

    def ols_predict(Xtrain, ytrain, Xeval):
        A = Xtrain.T @ Xtrain + ridge * np.eye(Xtrain.shape[1])
        b = Xtrain.T @ ytrain
        beta = np.linalg.solve(A, b)
        return Xeval @ beta

    pred_null = ols_predict(X_null_train, y_train, X_null_eval)
    pred_alt = ols_predict(X_alt_train, y_train, X_alt_eval)

    ll_null = gaussian_log_lik(y_eval - pred_null)
    ll_alt = gaussian_log_lik(y_eval - pred_alt)

    print(f"  [{label}] ll_alt={ll_alt:.2f}  ll_null={ll_null:.2f}  "
          f"log_e_contrib={ll_alt - ll_null:.3f}", flush=True)
    return ll_alt, ll_null


def run_per_kind_test(
    Z: np.ndarray, sides: np.ndarray, kinds: np.ndarray, ckpt: str
) -> dict[str, dict]:
    """
    For each entity kind, test whether the exp/noexp split is real structure.
    Uses split_likelihood_log_e (one-shot e-value from a single 50/50 split).
    Returns {kind: {log_e, verdict}}.
    """
    import gamfit

    results = {}
    for kind in sorted(set(kinds)):
        mask = kinds == kind
        if mask.sum() < 10:
            continue
        Zk = Z[mask]
        sk = sides[mask]
        n = len(Zk)
        half = n // 2
        rng = np.random.default_rng(RANDOM_STATE)
        perm = rng.permutation(n)
        train_idx = perm[:half]
        eval_idx = perm[half:]

        ll_alt, ll_null = fit_models_and_log_liks(
            Zk[train_idx], Zk[eval_idx],
            sk[train_idx], sk[eval_idx],
            label=f"{ckpt}/{kind}",
        )
        log_e = gamfit.split_likelihood_log_e(ll_alt, ll_null)
        results[kind] = dict(log_e=float(log_e), n=int(mask.sum()),
                             log_lik_alt=float(ll_alt), log_lik_null=float(ll_null))
    return results


def main() -> None:
    import gamfit

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cumulative gate across all checkpoints (anytime-valid)
    gate = gamfit.atom_birth_gate(ALPHA)

    all_results = {}
    cumulative_log_e = []

    for ckpt in CHECKPOINTS:
        ckpt_path = DATA_DIR / ckpt
        if not ckpt_path.exists():
            print(f"[SKIP] {ckpt}", flush=True)
            continue

        print(f"\n=== {ckpt} ===", flush=True)
        acts = np.load(ckpt_path / "activations.npy", mmap_mode="r")
        X_raw = np.array(acts[:, 25, :], dtype=np.float32)
        with open(ckpt_path / "prompts.jsonl") as fh:
            prompts = [json.loads(l) for l in fh]
        sides = np.array([p.get("side", "-") for p in prompts])
        kinds = np.array([p.get("kind", "?") for p in prompts])

        Z = pca_whiten(X_raw, PCA_DIM)
        n = len(Z)

        # 50/50 split for this checkpoint's global test
        rng = np.random.default_rng(RANDOM_STATE)
        perm = rng.permutation(n)
        half = n // 2
        train_idx, eval_idx = perm[:half], perm[half:]

        ll_alt, ll_null = fit_models_and_log_liks(
            Z[train_idx], Z[eval_idx],
            sides[train_idx], sides[eval_idx],
            label=f"{ckpt}/global",
        )

        # Feed into cumulative gate (shard = this checkpoint)
        gate.absorb_shard(ll_alt, ll_null)
        log_e = float(gate.log_e_value)
        verdict = str(gate.verdict)
        certified = bool(gate.certified)
        cumulative_log_e.append(dict(
            ckpt=ckpt, log_e=log_e, verdict=verdict, certified=certified,
            ll_alt=float(ll_alt), ll_null=float(ll_null),
        ))
        print(f"  [{ckpt}] cumulative log_e={log_e:.3f}  verdict={verdict}  "
              f"certified={certified}", flush=True)

        # Per-kind tests
        kind_results = run_per_kind_test(Z, sides, kinds, ckpt)

        # e-BH certificate across kinds
        kind_names = list(kind_results.keys())
        log_e_vals = [kind_results[k]["log_e"] for k in kind_names]
        if log_e_vals:
            certified_idx = gamfit.e_bh_dictionary_certificate(log_e_vals, ALPHA)
            certified_kinds = [kind_names[i] for i in certified_idx]
        else:
            certified_kinds = []

        print(f"  [{ckpt}] e-BH certified kinds: {certified_kinds}", flush=True)
        all_results[ckpt] = dict(
            kind_results=kind_results,
            certified_kinds=certified_kinds,
            global_log_e=log_e,
        )

    # Save JSON
    output = dict(cumulative_gate=cumulative_log_e, per_checkpoint=all_results)
    json_path = OUT_DIR / "structure_discovery.json"
    with open(json_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nSAVED {json_path}", flush=True)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpts_done = [r["ckpt"] for r in cumulative_log_e]
    log_es = [r["log_e"] for r in cumulative_log_e]
    certifieds = [r["certified"] for r in cumulative_log_e]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: cumulative log e-value
    ax = axes[0]
    xs = range(len(ckpts_done))
    ax.plot(xs, log_es, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.axhline(np.log(1 / ALPHA), color="#E91E63", linestyle="--",
               linewidth=1.5, label=f"threshold log(1/α)={np.log(1/ALPHA):.2f}")
    for i, (x, y, c) in enumerate(zip(xs, log_es, certifieds)):
        marker = "★" if c else "○"
        ax.annotate(marker, (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=12)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(ckpts_done, rotation=20, ha="right", fontsize=8)
    ax.set_xlabel("Checkpoint (sequential)")
    ax.set_ylabel("Cumulative log e-value")
    ax.set_title("Anytime-valid exp/noexp structure test\n(optional-stopping immune)", fontweight="bold")
    ax.legend(fontsize=8)

    # Right: per-kind log e-values at final checkpoint (stage3 or latest)
    ax2 = axes[1]
    final_ckpt = ckpts_done[-1] if ckpts_done else None
    if final_ckpt and final_ckpt in all_results:
        kr = all_results[final_ckpt]["kind_results"]
        cert_kinds = set(all_results[final_ckpt]["certified_kinds"])
        kind_names = sorted(kr.keys(), key=lambda k: -kr[k]["log_e"])
        log_e_vals = [kr[k]["log_e"] for k in kind_names]
        colors = ["#E91E63" if k in cert_kinds else "#90A4AE" for k in kind_names]
        ax2.barh(range(len(kind_names)), log_e_vals, color=colors)
        ax2.set_yticks(range(len(kind_names)))
        ax2.set_yticklabels(kind_names, fontsize=8)
        ax2.axvline(np.log(1 / ALPHA), color="#2196F3", linestyle="--",
                    linewidth=1.5, label=f"threshold")
        ax2.set_xlabel("Log e-value")
        ax2.set_title(f"Per entity-kind structure evidence\n({final_ckpt})", fontweight="bold")
        ax2.legend(fontsize=8)
        cert_patch = plt.matplotlib.patches.Patch(color="#E91E63", label="e-BH certified")
        uncert_patch = plt.matplotlib.patches.Patch(color="#90A4AE", label="not certified")
        ax2.legend(handles=[cert_patch, uncert_patch], fontsize=8)

    fig.suptitle(
        "OLMo-3-32B — L25 qualia structure discovery via e-processes\n"
        "Universal inference (split-LR); anytime-valid, optional-stopping safe",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "structure_discovery.png"
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)
    print(f"SAVED {out_path}", flush=True)

    # Print final verdict
    print("\n=== STRUCTURE DISCOVERY SUMMARY ===")
    for r in cumulative_log_e:
        star = " *** CERTIFIED ***" if r["certified"] else ""
        print(f"  {r['ckpt']:<28} log_e={r['log_e']:>7.3f}  verdict={r['verdict']}{star}")
    print()


if __name__ == "__main__":
    main()
