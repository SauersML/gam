#!/usr/bin/env python
"""K-ladder reconstruction-parity experiment for Qwen L12 activations.

This driver is intentionally serialized: every ``gamfit.sae_manifold_fit``
candidate runs in its own child process, one at a time, with wall-time and RSS
telemetry recorded into the JSON report. The parent writes partial results
after every candidate so solver failures and watchdog exits remain auditable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ACTIVATIONS = "/mnt/work/exp/qwen35_2b_wikitext_l12.pt"
DEFAULT_BASELINES = "/mnt/work/exp/qwen35_k8.json"
DEFAULT_OUT = "/mnt/work/exp/k_ladder.json"
DEFAULT_FIG = "/mnt/work/exp/figI_k_ladder.png"
MIXES = ("circle-only", "euclidean-only", "mixed")


def load_activations(
    path: str,
    *,
    tensor_key: str | None,
    n_rows: int | None,
    shuffle_seed: int | None,
) -> torch.Tensor:
    import torch

    if path.endswith(".npy"):
        x = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, torch.Tensor):
            x = payload
        else:
            x = payload[tensor_key or "X"]
    if x.ndim != 2:
        raise ValueError(f"activation matrix must be 2D; got shape {tuple(x.shape)}")
    x = x.float()
    if shuffle_seed is not None:
        generator = torch.Generator().manual_seed(int(shuffle_seed))
        x = x[torch.randperm(x.shape[0], generator=generator)]
    if n_rows is not None:
        n = int(n_rows)
        if n < 2:
            raise ValueError("--n must be at least 2")
        x = x[:n]
    return x


def deterministic_split(n_rows: int, eval_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("--eval-frac must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n_rows))
    n_eval = max(1, int(round(n_rows * float(eval_fraction))))
    n_eval = min(n_eval, n_rows - 1)
    eval_idx = np.sort(order[:n_eval])
    fit_idx = np.sort(order[n_eval:])
    return fit_idx, eval_idx


def normalize_from_fit(
    x_fit_raw: torch.Tensor,
    x_eval_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x_fit_raw.mean(0, keepdim=True)
    sigma = x_fit_raw.std(0, keepdim=True).clamp(min=1e-6)
    return (x_fit_raw - mu) / sigma, (x_eval_raw - mu) / sigma, mu, sigma


def explained_variance(x: np.ndarray, xhat: np.ndarray) -> float:
    num = float(np.square(x - xhat).sum())
    den = float(np.square(x - x.mean(axis=0, keepdims=True)).sum())
    return 1.0 - num / den if den > 0.0 else 0.0


def fit_dispersion(x_fit: np.ndarray, xhat_fit: np.ndarray) -> float:
    return max(float(np.square(x_fit - xhat_fit).mean()), 1e-12)


def gaussian_log_density_per_scalar(x: np.ndarray, xhat: np.ndarray, dispersion: float) -> float:
    mse = float(np.square(x - xhat).mean())
    return -0.5 * (math.log(2.0 * math.pi * float(dispersion)) + mse / float(dispersion))


def active_threshold(assignment: str, k_atoms: int, jumprelu_threshold: float = 0.0) -> float:
    if assignment == "softmax":
        return 1.0 / max(1, int(k_atoms))
    if assignment == "jumprelu":
        return float(jumprelu_threshold)
    return 0.5


def topology_mix(mix: str, k_atoms: int) -> tuple[list[str], list[int]]:
    if mix == "circle-only":
        return ["periodic"] * int(k_atoms), [1] * int(k_atoms)
    if mix == "euclidean-only":
        return ["euclidean"] * int(k_atoms), [1] * int(k_atoms)
    if mix == "mixed":
        bases = ["periodic" if i % 2 == 0 else "euclidean" for i in range(int(k_atoms))]
        return bases, [1] * int(k_atoms)
    raise ValueError(f"unknown topology mix {mix!r}; expected one of {MIXES}")


def parameter_count(model: Any) -> tuple[int, dict[str, int]]:
    decoder = int(sum(np.asarray(block).size for block in model.decoder_blocks))
    coords = int(sum(np.asarray(coord).size for coord in model.coords))
    assignments = int(np.asarray(model.assignments).size)
    logits = int(np.asarray(getattr(model, "low_level_logits", np.zeros((0, 0)))).size)
    total = decoder + coords + assignments + logits
    return total, {
        "decoder_coefficients": decoder,
        "fit_coordinates": coords,
        "fit_assignments": assignments,
        "fit_logits": logits,
    }


def atom_active_dims(model: Any, declared_dims: list[int]) -> list[int]:
    dims: list[int] = []
    for idx, atom in enumerate(model.atoms):
        value = getattr(atom, "active_dim", None)
        if value is None and idx < len(model.coords):
            value = np.asarray(model.coords[idx]).shape[1]
        if value is None:
            value = declared_dims[idx]
        dims.append(int(value))
    return dims


def active_accounting(
    assignments_eval: np.ndarray,
    dims: list[int],
    assignment: str,
    jumprelu_threshold: float,
) -> dict[str, Any]:
    threshold = active_threshold(assignment, assignments_eval.shape[1], jumprelu_threshold)
    active = assignments_eval > threshold
    charges = np.asarray([1 + int(d) for d in dims], dtype=np.float64)
    alive = active.sum(axis=0) > 0
    per_row_atoms = active.sum(axis=1)
    per_row_channels = (active * charges.reshape(1, -1)).sum(axis=1)
    return {
        "active_threshold": float(threshold),
        "active_scalar_channels_per_row": float(per_row_channels.mean()),
        "mean_active_atoms_per_eval_row": float(per_row_atoms.mean()),
        "alive_atoms_eval": int(alive.sum()),
        "alive_scalar_channels_eval": int(charges[alive].sum()),
        "atom_active_dims": [int(d) for d in dims],
    }


def run_fit_worker(args: argparse.Namespace) -> int:
    result: dict[str, Any] = {
        "method": "gam_manifold_sae",
        "K": int(args.K),
        "mix": args.mix,
        "assignment": args.assignment,
        "n_iter": int(args.n_iter),
        "seed": int(args.seed),
        "status": "error",
    }
    started = time.time()
    try:
        import gamfit
        import torch

        bases, dims = topology_mix(args.mix, args.K)
        result["atom_basis"] = bases
        result["declared_d_atom"] = dims
        x_raw = load_activations(
            args.acts,
            tensor_key=args.tensor_key,
            n_rows=args.n,
            shuffle_seed=args.shuffle_seed,
        )
        fit_idx, eval_idx = deterministic_split(x_raw.shape[0], args.eval_frac, args.split_seed)
        x_fit_raw = x_raw[torch.as_tensor(fit_idx)]
        x_eval_raw = x_raw[torch.as_tensor(eval_idx)]
        x_fit, x_eval, mu, sigma = normalize_from_fit(x_fit_raw, x_eval_raw)
        x_fit64 = x_fit.double().numpy()
        x_eval64 = x_eval.double().numpy()
        result["data"] = {
            "acts": args.acts,
            "n_total": int(x_raw.shape[0]),
            "p": int(x_raw.shape[1]),
            "fit_n": int(fit_idx.size),
            "eval_n": int(eval_idx.size),
            "eval_fraction": float(args.eval_frac),
            "split_seed": int(args.split_seed),
            "shuffle_seed": None if args.shuffle_seed is None else int(args.shuffle_seed),
            "normalization": "mean/std fit on fit split only and reused for eval",
        }

        model = gamfit.sae_manifold_fit(
            x_fit64,
            K=int(args.K),
            atom_basis=bases,
            d_atom=dims,
            assignment=args.assignment,
            n_iter=int(args.n_iter),
            random_state=int(args.seed),
            top_k=args.top_k,
        )
        latents = model.converged_latents(x_eval64)
        xhat_fit = np.asarray(model.fitted, dtype=np.float64)
        xhat_eval = np.asarray(latents["fitted"], dtype=np.float64)
        assignments_eval = np.asarray(latents["assignments"], dtype=np.float64)
        params, breakdown = parameter_count(model)
        active = active_accounting(
            assignments_eval,
            atom_active_dims(model, dims),
            model.assignment,
            float(getattr(model, "jumprelu_threshold", 0.0)),
        )
        dispersion = fit_dispersion(x_fit64, xhat_fit)
        x_eval_raw64 = x_eval_raw.numpy().astype(np.float64)
        mu64 = mu.numpy().astype(np.float64)
        sigma64 = sigma.numpy().astype(np.float64)
        result.update(
            {
                "status": "ok",
                "seconds": float(time.time() - started),
                "basis_specs": list(model.basis_specs),
                "atom_topology": str(model.atom_topology),
                "atom_topologies": list(model.atom_topologies),
                "reml_score_fit": float(model.reml_score),
                "reconstruction_r2_fit": float(model.reconstruction_r2),
                "eval_ev": explained_variance(x_eval64, xhat_eval),
                "eval_mse": float(np.square(x_eval64 - xhat_eval).mean()),
                "eval_ev_raw": explained_variance(x_eval_raw64, xhat_eval * sigma64 + mu64),
                "fit_dispersion": float(dispersion),
                "eval_log_density_per_scalar": gaussian_log_density_per_scalar(
                    x_eval64,
                    xhat_eval,
                    dispersion,
                ),
                "total_parameter_count": int(params),
                "parameter_count_breakdown": breakdown,
                "atoms": [
                    {
                        "atom": int(i),
                        "basis": str(getattr(atom, "basis", model.basis_specs[i])),
                        "active_dim": int(active["atom_active_dims"][i]),
                        "eval_assignment_mass": float(assignments_eval[:, i].sum()),
                        "eval_assignment_mean": float(assignments_eval[:, i].mean()),
                    }
                    for i, atom in enumerate(model.atoms)
                ],
                **active,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "seconds": float(time.time() - started),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-6000:],
            }
        )
    Path(args.worker_result).write_text(json.dumps(result, indent=2, default=str))
    return 0


def rss_kib(pid: int) -> int | None:
    status = Path(f"/proc/{int(pid)}/status")
    if not status.exists():
        return None
    try:
        text = status.read_text(errors="replace")
    except FileNotFoundError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1])
    return None


def candidate_id(k_atoms: int, mix: str) -> str:
    return f"K{k_atoms}_{mix.replace('-', '_')}"


def run_candidate_process(args: argparse.Namespace, k_atoms: int, mix: str) -> dict[str, Any]:
    out_path = Path(args.out)
    log_dir = Path(args.log_dir) if args.log_dir else out_path.with_suffix("").parent / "k_ladder_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cid = candidate_id(k_atoms, mix)
    result_file = Path(tempfile.gettempdir()) / f"{cid}_{os.getpid()}_{time.time_ns()}.json"
    stdout_log = log_dir / f"{cid}.log"
    rss_log = log_dir / f"{cid}.rss.jsonl"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--acts",
        args.acts,
        "--baseline-json",
        args.baseline_json,
        "--out",
        args.out,
        "--fig",
        args.fig,
        "--K",
        str(k_atoms),
        "--mix",
        mix,
        "--assignment",
        args.assignment,
        "--n-iter",
        str(args.n_iter),
        "--seed",
        str(args.seed),
        "--split-seed",
        str(args.split_seed),
        "--eval-frac",
        str(args.eval_frac),
        "--worker-result",
        str(result_file),
    ]
    if args.tensor_key is not None:
        cmd.extend(["--tensor-key", args.tensor_key])
    if args.n is not None:
        cmd.extend(["--n", str(args.n)])
    if args.shuffle_seed is not None:
        cmd.extend(["--shuffle-seed", str(args.shuffle_seed)])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])

    started = time.time()
    peak_kib = 0
    samples = 0
    with stdout_log.open("w") as log_f, rss_log.open("w") as rss_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        timed_out = False
        while proc.poll() is None:
            elapsed = time.time() - started
            current = rss_kib(proc.pid)
            if current is not None:
                peak_kib = max(peak_kib, current)
                rss_f.write(json.dumps({"elapsed": elapsed, "rss_kib": current}) + "\n")
                rss_f.flush()
                samples += 1
            if elapsed > float(args.wall_timeout):
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(float(args.rss_interval))
        return_code = proc.wait()

    if timed_out:
        return {
            "method": "gam_manifold_sae",
            "K": int(k_atoms),
            "mix": mix,
            "assignment": args.assignment,
            "n_iter": int(args.n_iter),
            "seed": int(args.seed),
            "status": "timeout",
            "seconds": float(time.time() - started),
            "wall_timeout_seconds": float(args.wall_timeout),
            "peak_rss_kib": int(peak_kib),
            "rss_samples": int(samples),
            "stdout_log": str(stdout_log),
            "rss_log": str(rss_log),
        }

    if result_file.exists():
        result = json.loads(result_file.read_text())
        result_file.unlink(missing_ok=True)
    else:
        result = {
            "method": "gam_manifold_sae",
            "K": int(k_atoms),
            "mix": mix,
            "assignment": args.assignment,
            "n_iter": int(args.n_iter),
            "seed": int(args.seed),
            "status": "process_error",
            "return_code": int(return_code),
            "seconds": float(time.time() - started),
        }
    result["return_code"] = int(return_code)
    result["peak_rss_kib"] = int(peak_kib)
    result["peak_rss_gib"] = float(peak_kib / 1024.0 / 1024.0)
    result["rss_samples"] = int(samples)
    result["stdout_log"] = str(stdout_log)
    result["rss_log"] = str(rss_log)
    return result


def all_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for child in value.values():
            found.extend(all_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(all_dicts(child))
    return found


def numeric_field(entry: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = entry.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def reference_from_entry(entry: dict[str, Any], label: str, source: str) -> dict[str, Any]:
    return {
        "method": "banked_reference",
        "label": label,
        "source": source,
        "eval_ev": numeric_field(entry, ("eval_ev", "heldout_ev", "held_out_ev", "ev")),
        "active_scalar_channels_per_row": numeric_field(
            entry,
            ("active_scalar_channels_per_row", "active_channels", "k", "rank", "top_k"),
        ),
        "total_parameter_count": numeric_field(
            entry,
            ("total_parameter_count", "parameter_count", "params", "n_params"),
        ),
        "raw": entry,
    }


def load_banked_references(path: str) -> list[dict[str, Any]]:
    source = str(path)
    data = json.loads(Path(path).read_text())
    entries = all_dicts(data)
    references: list[dict[str, Any]] = []

    def text(entry: dict[str, Any]) -> str:
        fields = [
            entry.get("method"),
            entry.get("label"),
            entry.get("role"),
            entry.get("name"),
            entry.get("topology"),
        ]
        return " ".join(str(v).lower() for v in fields if v is not None)

    official = [
        e for e in entries
        if "official" in text(e) and numeric_field(e, ("eval_ev", "heldout_ev", "held_out_ev", "ev")) is not None
    ]
    pca = [
        e for e in entries
        if "pca" in text(e) and int(round(numeric_field(e, ("rank", "active_scalar_channels_per_row")) or -1)) == 100
    ]
    vanilla = [
        e for e in entries
        if "vanilla" in text(e) and numeric_field(e, ("eval_ev", "heldout_ev", "held_out_ev", "ev")) is not None
    ]
    if official:
        references.append(reference_from_entry(max(official, key=lambda e: numeric_field(e, ("eval_ev", "ev")) or -1.0), "official_topk_k100", source))
    if pca:
        references.append(reference_from_entry(max(pca, key=lambda e: numeric_field(e, ("eval_ev", "ev")) or -1.0), "pca_rank100", source))
    if vanilla:
        references.append(reference_from_entry(max(vanilla, key=lambda e: numeric_field(e, ("eval_ev", "ev")) or -1.0), "vanilla_topk", source))
    return references


def comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for section in ("candidates", "references"):
        for entry in report.get(section, []):
            if entry.get("status", "ok") != "ok" and section == "candidates":
                continue
            rows.append(
                {
                    "section": section,
                    "label": entry.get("label") or candidate_id(entry.get("K", 0), entry.get("mix", "unknown")),
                    "K": entry.get("K"),
                    "mix": entry.get("mix"),
                    "eval_ev": entry.get("eval_ev"),
                    "eval_ev_raw": entry.get("eval_ev_raw"),
                    "active_scalar_channels_per_row": entry.get("active_scalar_channels_per_row"),
                    "total_parameter_count": entry.get("total_parameter_count"),
                    "alive_atoms_eval": entry.get("alive_atoms_eval"),
                    "mean_active_atoms_per_eval_row": entry.get("mean_active_atoms_per_eval_row"),
                }
            )
    return sorted(rows, key=lambda row: (str(row["section"]), str(row["label"])))


def write_report(report: dict[str, Any], path: str) -> None:
    report["comparison_table"] = comparison_rows(report)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, default=str))


def finite_positive(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0.0 else None


def plot_figI(report: dict[str, Any], fig_path: str) -> None:
    import matplotlib.pyplot as plt

    rows = comparison_rows(report)
    ok_rows = [row for row in rows if row.get("eval_ev") is not None]
    if not ok_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True, constrained_layout=True)
    mix_markers = {
        "circle-only": "o",
        "euclidean-only": "s",
        "mixed": "^",
        None: "D",
    }
    colors = {
        "circle-only": "#0072B2",
        "euclidean-only": "#D55E00",
        "mixed": "#009E73",
        None: "#4C4C4C",
    }
    for row in ok_rows:
        active = finite_positive(row.get("active_scalar_channels_per_row"))
        params = finite_positive(row.get("total_parameter_count"))
        ev = float(row["eval_ev"])
        mix = row.get("mix")
        marker = mix_markers.get(mix, "D")
        color = colors.get(mix, "#4C4C4C")
        size = 42.0 if active is None else 36.0 + 2.4 * min(active, 120.0)
        label = str(row.get("label"))
        if active is not None:
            axes[0].scatter(active, ev, s=size, marker=marker, color=color, alpha=0.82)
            axes[0].annotate(label, (active, ev), xytext=(4, 3), textcoords="offset points", fontsize=7)
        if params is not None:
            axes[1].scatter(params, ev, s=size, marker=marker, color=color, alpha=0.82)
            axes[1].annotate(label, (params, ev), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("active scalar channels per eval row")
    axes[1].set_xlabel("total parameter count")
    axes[0].set_ylabel("held-out explained variance")
    axes[0].set_title("EV vs active budget")
    axes[1].set_title("EV vs parameter count")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one K")
    return values


def parse_mix_list(raw: str) -> list[str]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one topology mix")
    for value in values:
        if value not in MIXES:
            raise ValueError(f"unknown mix {value!r}; expected one of {MIXES}")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--tensor-key", default=None)
    parser.add_argument("--baseline-json", default=DEFAULT_BASELINES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--fig", default=DEFAULT_FIG)
    parser.add_argument("--ks", default="8,16,32")
    parser.add_argument("--mixes", default="circle-only,euclidean-only,mixed")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--eval-frac", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=1026)
    parser.add_argument("--seed", type=int, default=1026)
    parser.add_argument("--assignment", default="ibp_map")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--wall-timeout", type=float, default=7200.0)
    parser.add_argument("--rss-interval", type=float, default=10.0)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--mix", default="circle-only")
    parser.add_argument("--worker-result", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.worker:
        if args.worker_result is None:
            raise SystemExit("--worker requires --worker-result")
        raise SystemExit(run_fit_worker(args))

    references = load_banked_references(args.baseline_json)
    report: dict[str, Any] = {
        "protocol": {
            "name": "issue-1026 K-ladder reconstruction-parity experiment",
            "acts": args.acts,
            "baseline_json": args.baseline_json,
            "ks": parse_int_list(args.ks),
            "mixes": parse_mix_list(args.mixes),
            "serialized_candidates": True,
            "wall_timeout_seconds": float(args.wall_timeout),
            "rss_interval_seconds": float(args.rss_interval),
            "assignment": args.assignment,
            "top_k": args.top_k,
            "n_iter": int(args.n_iter),
            "seed": int(args.seed),
            "split_seed": int(args.split_seed),
            "eval_fraction": float(args.eval_frac),
            "n_limit": args.n,
            "capacity": "active scalar channels per eval row from eval assignments; charge=1+active_dim per active atom",
            "parameter_count": "decoder coefficients + fitted coordinates + fitted assignments + logits from the fit payload",
        },
        "references": references,
        "candidates": [],
    }
    write_report(report, args.out)

    for k_atoms in report["protocol"]["ks"]:
        for mix in report["protocol"]["mixes"]:
            print(f"[candidate] K={k_atoms} mix={mix}", flush=True)
            result = run_candidate_process(args, int(k_atoms), str(mix))
            print(
                "[candidate] "
                f"K={k_atoms} mix={mix} status={result.get('status')} "
                f"ev={result.get('eval_ev')} peak_rss_gib={result.get('peak_rss_gib')}",
                flush=True,
            )
            report["candidates"].append(result)
            write_report(report, args.out)
            plot_figI(report, args.fig)

    plot_figI(report, args.fig)
    write_report(report, args.out)
    print(f"[done] wrote {args.out} and {args.fig}", flush=True)


if __name__ == "__main__":
    main()
