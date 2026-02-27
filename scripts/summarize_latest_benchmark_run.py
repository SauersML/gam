#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

DEFAULT_WORKFLOW = "benchmark.yml"
DEFAULT_OUT = Path("scripts/latest_benchmark_summary.md")
DEFAULT_LOCAL_TZ = "America/Chicago"


def run_cmd(args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=False, text=True, capture_output=capture)


def gh_api_json(path: str) -> Any:
    proc = run_cmd(["gh", "api", path])
    if proc.returncode != 0:
        raise RuntimeError(f"gh api failed for {path}: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)


def parse_owner_repo() -> tuple[str, str]:
    proc = run_cmd(["git", "remote", "get-url", "origin"])
    if proc.returncode != 0:
        raise RuntimeError("failed to read git origin URL")
    url = proc.stdout.strip()
    if url.startswith("git@"):
        # git@github.com:owner/repo.git
        repo = url.split(":", 1)[1]
    else:
        p = urlparse(url)
        repo = p.path.lstrip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    if "/" not in repo:
        raise RuntimeError(f"cannot parse owner/repo from origin URL: {url}")
    owner, name = repo.split("/", 1)
    return owner, name


def get_latest_workflow_run(owner: str, repo: str, workflow: str) -> dict[str, Any]:
    path = f"/repos/{owner}/{repo}/actions/workflows/{workflow}/runs?per_page=1"
    payload = gh_api_json(path)
    runs = payload.get("workflow_runs", [])
    if not runs:
        raise RuntimeError(f"no runs found for workflow '{workflow}'")
    return runs[0]


def list_artifacts(owner: str, repo: str, run_id: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        path = f"/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts?per_page=100&page={page}"
        payload = gh_api_json(path)
        chunk = payload.get("artifacts", [])
        if not chunk:
            break
        out.extend(chunk)
        if len(chunk) < 100:
            break
        page += 1
    return out


def download_artifact_zip(owner: str, repo: str, artifact_id: int, out_zip: Path) -> None:
    path = f"/repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip"
    with out_zip.open("wb") as fh:
        proc = subprocess.run(["gh", "api", path], check=False, stdout=fh, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to download artifact {artifact_id}: {proc.stderr.decode().strip()}")


def rank_values(values: list[float], *, higher_is_better: bool) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=higher_is_better)
    ranks = [0] * len(values)
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    return ranks


def fmt_float(v: Any, digits: int = 6) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return ""


def fmt_secs(v: Any) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.3f}"
    except Exception:
        return ""


def scenario_metric_spec(family: str) -> list[tuple[str, str, bool]]:
    fam = str(family or "").lower()
    if fam == "binomial":
        return [
            ("AUC (↑ better)", "auc", True),
            ("Brier (↓ better)", "brier", False),
            ("LogLoss (↓ better)", "logloss", False),
        ]
    if fam == "survival":
        return [("C-index (↑ better)", "auc", True)]
    if fam == "gaussian":
        return [
            ("RMSE (↓ better)", "rmse", False),
            ("MAE (↓ better)", "mae", False),
            ("R2 (↑ better)", "r2", True),
        ]
    return []


def render_scenario_block(name: str, rows: list[dict[str, Any]], run_ts_utc: datetime, tz_name: str) -> str:
    family = str(rows[0].get("family", "unknown"))
    cv_n = rows[0].get("_cv_n_splits")
    cv_seed = rows[0].get("_cv_seed")
    cv_safe = rows[0].get("_cv_leakage_safe")

    metrics = scenario_metric_spec(family)
    ok_rows = [dict(r, _row_idx=i) for i, r in enumerate(rows) if str(r.get("status", "")) == "ok"]
    if not ok_rows:
        return ""

    # ranks per metric among rows with non-null value
    rank_maps: dict[str, dict[int, int]] = {}
    for _, key, higher in metrics:
        idx = [int(r["_row_idx"]) for r in ok_rows if r.get(key) is not None]
        vals = [float(r[key]) for r in ok_rows if r.get(key) is not None]
        if not vals:
            rank_maps[key] = {}
            continue
        ranks = rank_values(vals, higher_is_better=higher)
        rank_maps[key] = {row_idx: rank for row_idx, rank in zip(idx, ranks)}

    primary_key = metrics[0][1] if metrics else None
    ok_rows = sorted(
        ok_rows,
        key=lambda r: (
            rank_maps.get(primary_key, {}).get(int(r.get("_row_idx", -1)), 10**9)
            if primary_key
            else 0,
            str(r.get("contender", "")),
        ),
    )

    ts_local_txt = ""
    if ZoneInfo is not None:
        try:
            local_dt = run_ts_utc.astimezone(ZoneInfo(tz_name))
            ts_local_txt = f" ({local_dt.strftime('%Y-%m-%d %H:%M:%S')} {tz_name})"
        except Exception:
            ts_local_txt = ""

    out: list[str] = []
    out.append(f"**Scenario:** `{name}` ({family})")
    out.append(f"**CV:** {cv_n}-fold (seed={cv_seed}, leakage_safe={str(cv_safe).lower()})")
    out.append(
        f"**Run timestamp:** {run_ts_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC{ts_local_txt}"
    )
    out.append("")

    headers = ["Contender"]
    for label, _, _ in metrics:
        short = label.split(" ")[0]
        headers.extend([label, f"{short} rank"])
    headers.extend(["Fit (s)", "Predict (s)"])

    aligns = [":--------------"]
    for _ in metrics:
        aligns.extend(["-------------:", "-------:"])
    aligns.extend(["------:", "----------:"])

    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(aligns) + " |")

    for r in ok_rows:
        row_idx = int(r.get("_row_idx", -1))
        row_cells = [f"`{r.get('contender','')}`"]
        for _, key, _ in metrics:
            row_cells.append(fmt_float(r.get(key), 6))
            rk = rank_maps.get(key, {}).get(row_idx)
            row_cells.append(str(rk) if rk is not None else "")
        row_cells.append(fmt_secs(r.get("fit_sec")))
        row_cells.append(fmt_secs(r.get("predict_sec")))
        out.append("| " + " | ".join(row_cells) + " |")

    out.append("")
    out.append("**Model specs**")
    out.append("")
    for r in ok_rows:
        out.append(f"* `{r.get('contender','')}`: `{r.get('model_spec','')}`")
    out.append("")
    return "\n".join(out)


def main() -> int:
    owner, repo = parse_owner_repo()
    run = get_latest_workflow_run(owner, repo, DEFAULT_WORKFLOW)
    run_id = int(run["id"])

    created_at = datetime.fromisoformat(str(run["created_at"]).replace("Z", "+00:00")).astimezone(timezone.utc)
    run_url = run.get("html_url", "")
    run_status = run.get("status", "")
    run_conclusion = run.get("conclusion", "")

    artifacts = list_artifacts(owner, repo, run_id)
    shard_artifacts = [
        a
        for a in artifacts
        if str(a.get("name", "")).startswith("bench-")
        and str(a.get("name", "")) != "bench-runtime"
        and not bool(a.get("expired", False))
    ]

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="gha_bench_summary_") as td:
        tdp = Path(td)
        for a in shard_artifacts:
            aid = int(a["id"])
            zpath = tdp / f"{aid}.zip"
            try:
                download_artifact_zip(owner, repo, aid, zpath)
            except Exception:
                continue
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    json_members = [m for m in zf.namelist() if m.endswith(".json")]
                    for m in json_members:
                        payload = json.loads(zf.read(m).decode("utf-8"))
                        cv = payload.get("cv", {})
                        for r in payload.get("results", []):
                            rr = dict(r)
                            rr["_cv_n_splits"] = cv.get("n_splits")
                            rr["_cv_seed"] = cv.get("seed")
                            rr["_cv_leakage_safe"] = cv.get("leakage_safe")
                            rows.append(rr)
            except Exception:
                continue

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sn = str(r.get("scenario_name", "")).strip()
        if sn:
            by_scenario[sn].append(r)

    md: list[str] = []
    md.append("# Latest Benchmark Run (Partial)\n")
    md.append(f"- Workflow: `{DEFAULT_WORKFLOW}`")
    md.append(f"- Run ID: `{run_id}`")
    md.append(f"- Status: `{run_status}`")
    md.append(f"- Conclusion: `{run_conclusion}`")
    md.append(f"- URL: {run_url}")
    md.append(f"- Completed shard artifacts found: `{len(shard_artifacts)}`")
    md.append("")

    if not by_scenario:
        md.append("No completed shard result artifacts were found in the latest run yet.")
    else:
        for scenario in sorted(by_scenario):
            block = render_scenario_block(scenario, by_scenario[scenario], created_at, DEFAULT_LOCAL_TZ)
            if block:
                md.append(block)

    out_path = DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
