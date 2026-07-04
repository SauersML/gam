"""Step 3 (task #25): compose + certify + evaluate driver for the 32k tiered SAE.

Reads Step-1 (certified linear Tier-1) + Step-2 (surviving curved charts) artifacts,
composes  μ (per-rollout Tier-0)  +  certified linear atoms  +  surviving curved
charts  into a tiered reconstruction, evaluates on HELD-OUT rollouts (total EV,
per-atom held-out leave-one-out drop, κ-null turning-Θ certificates for the curved
survivors), and emits THE report: the CERTIFIED K — linear and curved counts,
separately; that number is the project's deliverable — plus EV and the margin
distribution, in one JSON + a human-readable summary.

Why the composition is Python and not a Rust ``TieredSaeFit`` binding: ``TieredSaeFit``
is a pure-Rust spine (crates/gam-sae/src/tiered/mod.rs) with no PyO3 binding, so the
canonical Python composition path is ``examples/compose_tiers.compose_charts_from_manifest``
over the block-seed manifest — which this driver consumes.

CONTRACT v1
  - venv: ``source $ROOT/gamfit_current_manifest.sh`` (this driver imports gamfit).
  - split: ``block_nursery.train_test_split`` VERBATIM (canonical), applied to the
    ROLLOUT ids — whole-rollout, leak-free. Copied here with provenance so gam carries
    no dependency on the Manifold-SAE/experiments prototype tree.
  - per-rollout demean: Tier-0 conditioning (declared; part of the model, not discarded).
  - CHART FAILURE = SKIP, never a linear lift: a chart that fails/times out at compose
    contributes a ZERO recon (no curved atom); T1's frame already owns the block's linear
    structure, so a linear lift would double-count. Failures are counted in
    certified_K.charts_failed_to_compose; certified_K.curved counts only CONVERGED charts.
  - atom ledger JSONL schema (one object per line):
        {atom_id, block_g, kind, d_eff, delta_deviance, charge, margin, kept}
    kind ∈ {"linear", "curved(d=<N>)"}.

Build/verify against stubs before Step-1/Step-2 land:
    python compose32k_step3.py --make-fixture /tmp/mini
    python compose32k_step3.py --root /tmp/mini --out /tmp/mini/step3_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

LEDGER_FIELDS = ("atom_id", "block_g", "kind", "d_eff", "delta_deviance",
                 "charge", "margin", "kept")


# --------------------------------------------------------------------------- #
# canonical rollout split — VERBATIM from Manifold-SAE/experiments/block_nursery.py:382
# (tier2 nursery's A/B split). Copied, not imported, to keep gam free of the
# experiments tree; applied to the ROLLOUT count (not rows) for leak-free splitting.
# --------------------------------------------------------------------------- #
def train_test_split(n: int, frac: float = 0.7, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(round(frac * n))
    return np.sort(perm[:k]), np.sort(perm[k:])


def rollout_split_masks(rollout_ids: np.ndarray, frac: float, seed: int):
    """Whole-rollout train/held-out row masks via the canonical split on the
    UNIQUE rollouts (correlated rows in one rollout never straddle the boundary)."""
    rollout_ids = np.asarray(rollout_ids)
    uniq = np.unique(rollout_ids)
    tr_pos, te_pos = train_test_split(len(uniq), frac=frac, seed=seed)
    train_rolls = set(uniq[tr_pos].tolist())
    train_mask = np.array([r in train_rolls for r in rollout_ids], dtype=bool)
    return train_mask, ~train_mask, uniq[tr_pos], uniq[te_pos]


# --------------------------------------------------------------------------- #
# Tier-0: per-rollout demean (conditioning)
# --------------------------------------------------------------------------- #
def per_rollout_means(Z: np.ndarray, rollout_ids: np.ndarray) -> np.ndarray:
    """Per-row Tier-0 mean μ (the row's own rollout mean). Conditioning: each
    rollout's mean is estimated from that rollout's rows and is part of the model."""
    Z = np.asarray(Z, dtype=np.float64)
    rollout_ids = np.asarray(rollout_ids)
    mu = np.zeros_like(Z)
    for r in np.unique(rollout_ids):
        m = rollout_ids == r
        mu[m] = Z[m].mean(axis=0, keepdims=True)
    return mu


def explained_variance_train_mean(x, recon, train_mean) -> float:
    """EV = 1 − SSE/TSS, TSS about the TRAIN-split column mean. Mirror of
    examples/compose_artifact_schema.explained_variance_train_mean."""
    x = np.asarray(x, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    m = np.asarray(train_mean, dtype=np.float64).reshape(1, -1)
    rss = float(np.sum((x - recon) ** 2))
    tss = float(np.sum((x - m) ** 2))
    return 1.0 - rss / tss if tss > 0.0 else 0.0


# --------------------------------------------------------------------------- #
# artifact IO
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def validate_ledger(rows: list[dict], where: str) -> None:
    for i, r in enumerate(rows):
        missing = [f for f in LEDGER_FIELDS if f not in r]
        if missing:
            raise ValueError(f"{where} row {i} missing ledger fields {missing}")


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
def linear_recon_per_atom(Z0: np.ndarray, decoder: np.ndarray) -> np.ndarray:
    """OOS least-squares linear codes on the CENTERED data, then per-atom recon.
    Returns (K, n, p): the reconstruction contributed by each linear atom, so the
    total is the sum over axis 0 and per-atom leave-one-out is a subtraction.
    decoder rows are the K atom directions in R^p."""
    Z0 = np.asarray(Z0, dtype=np.float64)
    D = np.asarray(decoder, dtype=np.float64)          # (K, p)
    if D.shape[0] == 0:
        return np.zeros((0,) + Z0.shape, dtype=np.float64)
    # least-squares codes C (n,K): argmin ||Z0 - C D||  ->  C = Z0 Dᵀ (D Dᵀ)⁻¹
    gram = D @ D.T
    codes = Z0 @ D.T @ np.linalg.pinv(gram)            # (n, K)
    return codes.T[:, :, None] * D[:, None, :]         # (K, n, p)


_FIT_WORKER = r'''
import inspect, json, os, sys, numpy as np, gamfit
# #28 driver-side heartbeat: pass harness_util.progress_heartbeat() as progress_callback
# so a long fit emits progress (no silent hang; the run always ends with a signal).
# Guarded: optional if harness_util or the kwarg isn't present (local/mini runs).
pcb = None
for _d in (os.environ.get("GAM_HARNESS_UTIL"),
           "/projects/standard/hsiehph/sauer354/scratch/compose32k/scripts"):
    if _d and os.path.isfile(os.path.join(_d, "harness_util.py")):
        sys.path.insert(0, _d); break
try:
    import harness_util
    pcb = harness_util.progress_heartbeat()
except Exception:
    pcb = None
_kw = {}
if pcb is not None and "progress_callback" in inspect.signature(gamfit.sae_manifold_fit).parameters:
    _kw["progress_callback"] = pcb
z = np.load(sys.argv[1])
chart = gamfit.sae_manifold_fit(np.ascontiguousarray(z, dtype=np.float32), K=1,
    d_atom=int(sys.argv[3]), atom_topology=sys.argv[4], n_iter=int(sys.argv[5]),
    random_state=int(sys.argv[6]), **_kw)
recon = np.asarray(chart.reconstruct(z), dtype=np.float64)
theta = None
try:
    hs = getattr(chart, "hybrid_split", None)
    if isinstance(hs, dict):
        v = hs.get("verdicts") or hs.get("atoms") or []
        if v and isinstance(v[0], dict):
            theta = v[0].get("fitted_turning")
except Exception:
    pass
np.save(sys.argv[2], recon)
print("THETA " + json.dumps(theta))
'''


def _isolated_fit(z: np.ndarray, *, d_atom: int, topology: str, n_iter: int,
                  random_state: int, wall_s: float):
    """Fit one K=1 chart in an ISOLATED subprocess with a wall-clock timeout, so a
    thrashing/co-collapsing fit (e.g. the #1026 oscillation on pre-recovery wheels,
    or any pathological block at 16384-scale) is KILLED and the caller falls back to
    a linear lift — it can never hang the driver. Returns (recon (n,p') | None, Θ | None).
    Pattern mirrors block_nursery.fit_curved_isolated."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        zin = str(Path(td) / "z.npy"); rout = str(Path(td) / "recon.npy")
        np.save(zin, np.ascontiguousarray(z, dtype=np.float32))
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _FIT_WORKER, zin, rout,
                 str(int(d_atom)), str(topology), str(int(n_iter)), str(int(random_state))],
                capture_output=True, text=True, timeout=wall_s)
        except subprocess.TimeoutExpired:
            return None, None
        if proc.returncode != 0 or not Path(rout).is_file():
            return None, None
        theta = None
        for line in proc.stdout.splitlines():
            if line.startswith("THETA "):
                try:
                    theta = json.loads(line[6:])
                except Exception:
                    theta = None
        return np.load(rout), theta


def curved_recon_per_block(prefix: str, *, chart_topology: str, chart_d_atom: int,
                           chart_n_iter: int, random_state: int, wall_s: float):
    """Per-block curved-chart reconstruction (the per-block-exposing form of the
    canonical examples/compose_tiers.compose_charts_from_manifest): fit one K=1 chart
    per block on the stored in-block coords, lift with the stored (p,b) basis Q.
    Returns (per_block_recon list of (n,p), report). Exposing per block is what lets
    the driver do per-atom held-out LOO; the summed recon is identical to the
    canonical composer's. Each chart fit is subprocess-isolated (hang-safe)."""
    import compose_tiers as ct  # lazy: pulls gamfit
    manifest, bases, coords = ct.load_seed_manifest(prefix)
    n_blocks = int(manifest["n_blocks"])
    p = int(manifest["ambient_p"])
    n_rows = int(next(iter(coords.values())).shape[0]) if coords else 0
    per_block_recon: list[np.ndarray] = []
    per_block_report: list[dict] = []
    for g in range(n_blocks):
        q = np.asarray(bases[g], dtype=np.float64)     # (p, b)
        z = np.asarray(coords[g], dtype=np.float64)    # (n, b)
        b = int(q.shape[1])
        rec: dict[str, Any] = {"block": g, "block_dim": b}
        z_hat, theta = _isolated_fit(z, d_atom=min(chart_d_atom, b), topology=chart_topology,
                                     n_iter=chart_n_iter, random_state=random_state, wall_s=wall_s)
        if z_hat is not None:
            rec["chart_status"] = "CONVERGED"
            rec["fitted_turning"] = theta
            per_block_recon.append(np.asarray(z_hat, dtype=np.float64) @ q.T)
        else:
            # SKIP on failure — contribute NOTHING (a ZERO recon), do NOT lift z@Qᵀ.
            # T1's frame ALREADY reconstructs this block's LINEAR structure; a linear
            # lift here would DOUBLE-COUNT the block's energy (inflating composed EV +
            # corrupting per-atom LOO). Honest semantics: a failed chart = no curved
            # atom for that block; T1 alone owns it (counted in report.charts_failed).
            rec["chart_status"] = "SKIPPED_NO_CURVED"
            rec["fitted_turning"] = None
            per_block_recon.append(np.zeros((n_rows, p), dtype=np.float64))
        per_block_report.append(rec)
    return per_block_recon, {"n_blocks": n_blocks, "ambient_p": p,
                             "per_block": per_block_report}


def kappa_null_theta(prefix: str, *, chart_topology: str, chart_d_atom: int,
                     chart_n_iter: int, seed: int, wall_s: float, n_draws: int = 4) -> dict:
    """Matched-Gaussian κ-null: refit a K=1 chart on Gaussian coords with the same
    per-block second moment; the q99 of the null turning Θ is the acceptance
    threshold (curved survives iff its Θ exceeds the null's). Lightweight v1 of the
    matched_null battery — enough to certify the mini-fixture survivors. Each null
    fit is subprocess-isolated (hang-safe)."""
    import compose_tiers as ct
    manifest, _bases, coords = ct.load_seed_manifest(prefix)
    rng = np.random.default_rng(seed)
    null_thetas: list[float] = []
    for g in range(int(manifest["n_blocks"])):
        z = np.asarray(coords[g], dtype=np.float64)
        cov = np.atleast_2d(np.cov(z, rowvar=False))
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(cov.shape[0]))
        for _ in range(n_draws):
            gauss = (rng.standard_normal(z.shape) @ L.T)
            _rec, t = _isolated_fit(gauss, d_atom=min(chart_d_atom, z.shape[1]),
                                    topology=chart_topology, n_iter=chart_n_iter,
                                    random_state=0, wall_s=wall_s)
            if t is not None:
                null_thetas.append(float(t))
    if not null_thetas:
        return {"theta_accept": None, "n_null": 0}
    return {"theta_accept": float(np.quantile(null_thetas, 0.99)),
            "n_null": len(null_thetas),
            "null_theta_mean": float(np.mean(null_thetas))}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(root: Path, out: Path, *, frac: float, seed: int,
        chart_topology: str, chart_d_atom: int, chart_n_iter: int, chart_wall_s: float) -> dict:
    root = Path(root)
    # --- load artifacts ---
    az = np.load(root / "activations.npz")
    Z = np.ascontiguousarray(az["Z"], dtype=np.float64)
    rollout_ids = np.asarray(az["rollout_ids"])
    n, p = Z.shape

    t1_ledger = read_jsonl(root / "t1" / "certified_atoms.jsonl")
    validate_ledger(t1_ledger, "t1 ledger")
    decoder = np.load(root / "t1" / "decoder.npy") if (root / "t1" / "decoder.npy").is_file() \
        else np.zeros((0, p), dtype=np.float32)
    charts_prefix = str(root / "charts" / "charts")
    charts_ledger = read_jsonl(root / "charts" / "certified_charts.jsonl")
    validate_ledger(charts_ledger, "charts ledger")

    kept_lin = [a for a in t1_ledger if a.get("kept")]
    kept_curved = [a for a in charts_ledger if a.get("kept")]

    # --- split rollouts (canonical, leak-free) ---
    train_mask, held_mask, train_rolls, held_rolls = rollout_split_masks(rollout_ids, frac, seed)
    train_mean = Z[train_mask].mean(axis=0)  # EV baseline (train-split column mean)

    # --- Tier-0: per-rollout demean (conditioning) ---
    mu = per_rollout_means(Z, rollout_ids)
    Z0 = Z - mu

    # --- Tier-1 linear: per-atom recon on the centered data, keep only kept atoms ---
    kept_lin_idx = [i for i, a in enumerate(t1_ledger) if a.get("kept")]
    D_kept = decoder[kept_lin_idx] if decoder.shape[0] else np.zeros((0, p))
    lin_per_atom = linear_recon_per_atom(Z0, D_kept)          # (Klin, n, p)
    lin_total = lin_per_atom.sum(axis=0) if lin_per_atom.shape[0] else np.zeros((n, p))

    # --- Tier-2 curved: per-block chart recon (canonical composer, per-block-exposed) ---
    curved_per_block, curved_report = curved_recon_per_block(
        charts_prefix, chart_topology=chart_topology, chart_d_atom=chart_d_atom,
        chart_n_iter=chart_n_iter, random_state=seed, wall_s=chart_wall_s)
    # keep only certified-survivor blocks whose chart ACTUALLY composed (CONVERGED).
    # A kept survivor whose chart failed/timed out at compose contributes NO curved atom
    # (skipped, T1 owns the block) and is counted in charts_failed — never linear-lifted.
    conv_blocks = {g for g, rec in enumerate(curved_report["per_block"])
                   if rec.get("chart_status") == "CONVERGED"}
    kept_curved_blocks = {int(a["block_g"]) for a in kept_curved}
    composed_blocks = kept_curved_blocks & conv_blocks
    charts_failed = sorted(kept_curved_blocks - conv_blocks)
    composed_curved = [a for a in kept_curved if int(a["block_g"]) in composed_blocks]
    curved_per_atom = [(g, r) for g, r in enumerate(curved_per_block) if g in composed_blocks]
    curved_total = np.sum([r for _, r in curved_per_atom], axis=0) if curved_per_atom else np.zeros((n, p))

    full_recon = mu + lin_total + curved_total

    # --- held-out EV (total) ---
    ev_full = explained_variance_train_mean(Z[held_mask], full_recon[held_mask], train_mean)
    ev_t0 = explained_variance_train_mean(Z[held_mask], mu[held_mask], train_mean)
    ev_t0_t1 = explained_variance_train_mean(Z[held_mask], (mu + lin_total)[held_mask], train_mean)

    # --- per-atom held-out LOO drop (ΔEV = EV_full − EV_without_atom) ---
    per_atom_report: list[dict] = []
    for j, a in enumerate(kept_lin):
        contrib = lin_per_atom[j]
        ev_wo = explained_variance_train_mean(Z[held_mask], (full_recon - contrib)[held_mask], train_mean)
        per_atom_report.append({**{k: a.get(k) for k in LEDGER_FIELDS},
                                "heldout_loo_drop": round(ev_full - ev_wo, 6)})
    for (g, contrib), a in zip(curved_per_atom, composed_curved):
        ev_wo = explained_variance_train_mean(Z[held_mask], (full_recon - contrib)[held_mask], train_mean)
        per_atom_report.append({**{k: a.get(k) for k in LEDGER_FIELDS},
                                "heldout_loo_drop": round(ev_full - ev_wo, 6)})

    # --- #29 HELD-OUT-VALIDATED certified-K (THE headline) ---
    # An atom counts toward the HEADLINE certified-K only if its held-out LOO contribution is
    # POSITIVE — the backstop against the two anti-conservative biases in train-side margins:
    # (a) token autocorrelation inflates train Δdeviance ∝n while the charge grows ∝log n, and
    # (b) the blocks are POST-SELECTION objects (T1 chose them on the same train data). The
    # train-margin count is kept as secondary.
    def _is_curved(row):
        return str(row.get("kind", "")).startswith("curved")
    val_lin = sum(1 for r in per_atom_report
                  if not _is_curved(r) and (r.get("heldout_loo_drop") or 0.0) > 0.0)
    val_curved = sum(1 for r in per_atom_report
                     if _is_curved(r) and (r.get("heldout_loo_drop") or 0.0) > 0.0)

    # --- κ-null Θ certificates for curved survivors ---
    null = kappa_null_theta(charts_prefix, chart_topology=chart_topology,
                            chart_d_atom=chart_d_atom, chart_n_iter=chart_n_iter,
                            seed=seed, wall_s=chart_wall_s)
    theta_accept = null.get("theta_accept")
    curved_certs = []
    for a in composed_curved:
        rec = curved_report["per_block"][int(a["block_g"])]
        theta = rec.get("fitted_turning")
        passes = (theta is not None and theta_accept is not None and theta > theta_accept)
        curved_certs.append({"atom_id": a.get("atom_id"), "block_g": a.get("block_g"),
                             "fitted_turning": theta, "theta_accept": theta_accept,
                             "kappa_null_pass": bool(passes)})

    margins = [a.get("margin") for a in (kept_lin + composed_curved) if a.get("margin") is not None]

    report = {
        "schema": "compose32k_step3.v1",
        "root": str(root),
        "n_rows": int(n), "p": int(p),
        "split": {"fn": "block_nursery.train_test_split (verbatim, per-rollout)",
                  "frac": frac, "seed": seed,
                  "n_rollouts": int(len(np.unique(rollout_ids))),
                  "n_train_rollouts": int(len(train_rolls)),
                  "n_held_rollouts": int(len(held_rolls)),
                  "n_held_rows": int(held_mask.sum())},
        "tier0": {"conditioning": "per-rollout demean"},
        # #29 caveat fields — MANDATORY, so the number is never read out of scope:
        "scope": "within-block curvature only (Mode-A blind spot: cross-block charts invisible)",
        "conditioning": "per-rollout μ (per-instance); slow features excluded (#21)",
        "certified_K": {
            # THE headline: held-out-validated (train-certified AND held-out LOO > 0).
            "headline": {"linear": val_lin, "curved": val_curved, "total": val_lin + val_curved,
                         "definition": "held-out-validated: train-certified AND held-out LOO contribution > 0"},
            # secondary — train-side margin count (anti-conservatively biased; see #29).
            "train_margin": {"linear": len(kept_lin), "curved": len(composed_curved),
                             "total": len(kept_lin) + len(composed_curved)},
            "curved_certified_in_step2": len(kept_curved),
            "charts_failed_to_compose": charts_failed},
        "heldout_ev": {"tier0_only": round(ev_t0, 6),
                       "tier0_tier1": round(ev_t0_t1, 6),
                       "full_composed": round(ev_full, 6),
                       "ev_contract": {"baseline": "train_mean",
                                       "definition": "1 - SSE/TSS",
                                       "eval_split": "heldout_rollouts"}},
        "margins": {"n": len(margins),
                    "min": round(float(np.min(margins)), 4) if margins else None,
                    "median": round(float(np.median(margins)), 4) if margins else None,
                    "max": round(float(np.max(margins)), 4) if margins else None},
        "kappa_null": null,
        "curved_certificates": curved_certs,
        "per_atom": per_atom_report,
        "curved_compose_report": curved_report,
    }
    out = Path(out)
    out.write_text(json.dumps(report, indent=2))
    _print_summary(report)
    return report


def _print_summary(r: dict) -> None:
    ck = r["certified_K"]; ev = r["heldout_ev"]; hd = ck["headline"]; tm = ck["train_margin"]
    print("\n================ COMPOSE 32k — STEP 3 REPORT ================", flush=True)
    print(f"  CERTIFIED K (held-out-validated):  linear={hd['linear']}  curved={hd['curved']}  "
          f"total={hd['total']}", flush=True)
    print(f"    (train-margin, secondary: linear={tm['linear']} curved={tm['curved']} total={tm['total']}; "
          f"charts_failed={len(ck['charts_failed_to_compose'])})", flush=True)
    print(f"  held-out EV:  T0={ev['tier0_only']:.4f}  ->  T0+T1={ev['tier0_tier1']:.4f}  "
          f"->  composed={ev['full_composed']:.4f}", flush=True)
    print(f"  split: {r['split']['n_train_rollouts']} train / {r['split']['n_held_rollouts']} "
          f"held rollouts ({r['split']['n_held_rows']} held rows)", flush=True)
    m = r["margins"]
    if m["n"]:
        print(f"  margins: min={m['min']}  median={m['median']}  max={m['max']}  (n={m['n']})", flush=True)
    nc = [c for c in r["curved_certificates"] if c["kappa_null_pass"]]
    print(f"  κ-null: {len(nc)}/{len(r['curved_certificates'])} curved survivors pass "
          f"(theta_accept={r['kappa_null'].get('theta_accept')})", flush=True)
    print("============================================================\n", flush=True)


# --------------------------------------------------------------------------- #
# synthetic mini-fixture (stub Step-1/Step-2 artifacts)
# --------------------------------------------------------------------------- #
def make_mini_fixture(outdir: Path, *, seed: int = 0) -> None:
    """Write a small synthetic fixture mimicking Step-1 (t1/) + Step-2 (charts/)
    outputs, so the driver runs green before the real artifacts land. Structure:
      <outdir>/activations.npz        Z (n,p) f32, rollout_ids (n,) int
      <outdir>/t1/certified_atoms.jsonl   linear ledger
      <outdir>/t1/decoder.npy             (Klin,p) linear decoder
      <outdir>/t1/tier1_result.json       fit metadata
      <outdir>/charts/charts.seeds.{json,npz}  curved manifest (block_seed_manifest.v1)
      <outdir>/charts/certified_charts.jsonl   curved ledger
    """
    outdir = Path(outdir)
    (outdir / "t1").mkdir(parents=True, exist_ok=True)
    (outdir / "charts").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_roll, per_roll, p = 8, 40, 16
    n = n_roll * per_roll
    rollout_ids = np.repeat(np.arange(n_roll), per_roll)

    # per-rollout mean (Tier-0 conditioning signal)
    roll_mean = rng.standard_normal((n_roll, p)).astype(np.float64) * 0.5
    # linear tier: 3 atoms
    klin = 3
    Dlin = rng.standard_normal((klin, p)); Dlin /= np.linalg.norm(Dlin, axis=1, keepdims=True)
    codes = np.zeros((n, klin))
    for i in range(n):
        act = rng.choice(klin, size=2, replace=False)
        codes[i, act] = rng.standard_normal(2)
    linear_part = codes @ Dlin
    # curved tier: 2 blocks, each a circle in its own 2-plane
    nblk, b = 2, 2
    Q = [np.linalg.qr(rng.standard_normal((p, b)))[0][:, :b] for _ in range(nblk)]
    theta = rng.uniform(0, 2 * np.pi, (n, nblk))
    curved_part = np.zeros((n, p))
    coords = {}
    for g in range(nblk):
        cz = np.stack([np.cos(theta[:, g]), np.sin(theta[:, g])], axis=1)  # (n,2)
        curved_part += cz @ Q[g].T
        coords[g] = cz.astype(np.float32)
    noise = 0.05 * rng.standard_normal((n, p))
    Z = (roll_mean[rollout_ids] + linear_part + curved_part + noise).astype(np.float32)
    np.savez(outdir / "activations.npz", Z=Z, rollout_ids=rollout_ids.astype(np.int64))

    # t1 artifacts
    np.save(outdir / "t1" / "decoder.npy", Dlin.astype(np.float32))
    t1_rows = [{"atom_id": f"lin{i}", "block_g": -1, "kind": "linear",
                "d_eff": 1.0, "delta_deviance": round(50.0 - 5 * i, 3),
                "charge": round(0.5 * np.log(n), 3), "margin": round(40.0 - 5 * i, 3),
                "kept": True} for i in range(klin)]
    write_jsonl(outdir / "t1" / "certified_atoms.jsonl", t1_rows)
    (outdir / "t1" / "tier1_result.json").write_text(json.dumps(
        {"schema": "tier1_result.stub", "K": klin, "P": p, "n": n,
         "certified_linear": klin}, indent=2))

    # charts manifest (block_seed_manifest.v1) + npz
    manifest = {"schema": "block_seed_manifest.v1", "n_blocks": nblk, "block_size": b,
                "block_topk": 1, "ambient_p": p, "gamma": 0.0,
                "explained_variance": 0.0, "residual_target": True, "n_basis_chart": 4,
                "blocks": [{"block": g, "block_dim": b} for g in range(nblk)]}
    (outdir / "charts" / "charts.seeds.json").write_text(json.dumps(manifest, indent=2))
    npz = {}
    for g in range(nblk):
        npz[f"block{g}_basis"] = Q[g].astype(np.float32)
        npz[f"block{g}_coords"] = coords[g]
    np.savez(outdir / "charts" / "charts.seeds.npz", **npz)
    chart_rows = [{"atom_id": f"curved{g}", "block_g": g, "kind": f"curved(d={b})",
                   "d_eff": round(2.0 * 1.2, 3), "delta_deviance": round(80.0 - 10 * g, 3),
                   "charge": round(0.5 * 2 * 1.2 * np.log(n), 3),
                   "margin": round(55.0 - 10 * g, 3), "kept": True} for g in range(nblk)]
    write_jsonl(outdir / "charts" / "certified_charts.jsonl", chart_rows)
    print(f"[fixture] wrote mini-fixture to {outdir}  (n={n} p={p} rollouts={n_roll} "
          f"linear={klin} curved-blocks={nblk})", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Step 3: compose+certify+evaluate driver (32k tiered SAE)")
    ap.add_argument("--make-fixture", type=str, default=None,
                    help="write a synthetic mini-fixture to this dir and exit")
    ap.add_argument("--root", type=str, default=None,
                    help="artifact root (activations.npz + t1/ + charts/)")
    ap.add_argument("--out", type=str, default=None, help="output report JSON")
    ap.add_argument("--frac", type=float, default=0.8,
                    help="train fraction of rollouts (Contract v1 canon = 0.8; all 4 steps split identically)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chart-topology", type=str, default="circle")
    ap.add_argument("--chart-d-atom", type=int, default=2)
    ap.add_argument("--chart-n-iter", type=int, default=50)
    ap.add_argument("--chart-wall-s", type=float, default=90.0,
                    help="per-chart subprocess wall-clock timeout (hang-safe)")
    return ap


def main() -> None:
    a = build_parser().parse_args()
    # let the driver find compose_tiers.py (its dir) regardless of CWD
    here = Path(__file__).resolve().parent
    for cand in (os.environ.get("GAM_EXAMPLES"), str(here)):
        if cand and (Path(cand) / "compose_tiers.py").is_file():
            sys.path.insert(0, cand)
            break
    if a.make_fixture:
        make_mini_fixture(Path(a.make_fixture), seed=a.seed)
        return
    if not a.root:
        raise SystemExit("need --root <artifact dir> (or --make-fixture <dir>)")
    out = a.out or str(Path(a.root) / "step3_report.json")
    run(Path(a.root), Path(out), frac=a.frac, seed=a.seed,
        chart_topology=a.chart_topology, chart_d_atom=a.chart_d_atom,
        chart_n_iter=a.chart_n_iter, chart_wall_s=a.chart_wall_s)


if __name__ == "__main__":
    main()
