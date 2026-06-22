import argparse
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile
import importlib.util

def validate_schemas():
    mod_path = pathlib.Path("bench/run_suite.py").resolve()
    spec = importlib.util.spec_from_file_location("run_suite_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = json.loads(pathlib.Path("bench/scenarios.json").read_text())
    scenarios = cfg.get("scenarios", [])
    if not scenarios:
        raise SystemExit("No benchmark scenarios found in bench/scenarios.json")

    for s in scenarios:
        mod.validate_scenario_schema(s)
    print(f"validated {len(scenarios)} scenario dataset schemas")

def validate_geo_subpop():
    mod_path = pathlib.Path("bench/run_suite.py").resolve()
    spec = importlib.util.spec_from_file_location("run_suite_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = json.loads(pathlib.Path("bench/scenarios.json").read_text())
    scenarios = cfg.get("scenarios", [])
    geo_subpop = [s for s in scenarios if str(s.get("name", "")).startswith("geo_subpop16_")]
    if not geo_subpop:
        raise SystemExit("No geo_subpop16 scenarios found in bench/scenarios.json")

    for s in geo_subpop:
        mod.validate_scenario_schema(s)
    print(f"validated geo_subpop16 simulation for {len(geo_subpop)} scenarios")

def build_matrix():
    SERIAL_SCENARIOS = {
        "icu_survival_death",
        "cirrhosis_survival",
    }

    def choose_single_knot(variants):
        by_k = {v["k"]: v for v in variants}
        if 12 in by_k:
            return by_k[12]["scenario"]
        ordered = sorted(variants, key=lambda v: v["k"])
        return ordered[len(ordered) // 2]["scenario"]

    cfg = json.loads(pathlib.Path("bench/scenarios.json").read_text())
    scenarios = cfg.get("scenarios", [])
    names = [s["name"] for s in scenarios if "name" in s]
    if not names:
        raise SystemExit("No benchmark scenarios found in bench/scenarios.json")

    event_name = os.environ.get("GITHUB_EVENT_NAME", "").strip().lower()
    is_nightly = event_name == "schedule"

    if is_nightly:
        selected = names
    else:
        pattern = re.compile(r"^(.*?)_n(\d+)_k(\d+)$")
        parsed = []
        for name in names:
            m = pattern.match(name)
            if not m:
                selected.append(name)
                continue
            family = m.group(1)
            n_val = int(m.group(2))
            k_val = int(m.group(3))
            parsed.append({
                "scenario": name,
                "family": family,
                "n": n_val,
                "k": k_val,
            })
        by_family_n = {}
        for p in parsed:
            key = (p["family"], p["n"])
            by_family_n.setdefault(key, []).append(p)
        
        selected = []
        for key, variants in by_family_n.items():
            selected.append(choose_single_knot(variants))
        
        from_unparsed = [n for n in names if not pattern.match(n)]
        selected.extend(from_unparsed)
        selected = sorted(list(set(selected)))

    matrix = {"scenario": selected}
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"matrix={json.dumps(matrix)}\n")
        f.write(f"is_nightly={'true' if is_nightly else 'false'}\n")

def extract_maturin_wheel(out_dir_arg="gamfit"):
    wheels = sorted(glob.glob("dist/*.whl"))
    if not wheels:
        sys.exit("maturin produced no wheel under dist/")
    wheel = wheels[-1]
    out_dir = pathlib.Path(out_dir_arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as zf:
        matches = [
            m for m in zf.namelist()
            if m.startswith("gamfit/_rust") and (m.endswith(".so") or m.endswith(".pyd"))
        ]
        if not matches:
            sys.exit(f"no _rust*.so found inside {wheel}")
        for member in matches:
            target = out_dir / pathlib.Path(member).name
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"extracted {target}")

def download_artifacts(target_name, out_dir_arg):
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    out_dir = pathlib.Path(out_dir_arg)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        ["gh", "api", f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"],
        check=True,
        capture_output=True,
        text=True,
    )
    artifacts = json.loads(proc.stdout).get("artifacts", [])
    if target_name == "bench-runtime":
        shard_artifacts = [a for a in artifacts if a["name"] == "bench-runtime"]
    else:
        shard_artifacts = [a for a in artifacts if a["name"].startswith(target_name)]
    
    if not shard_artifacts:
        print(f"no artifacts matching {target_name}")
        return

    for a in shard_artifacts:
        subprocess.run(
            ["gh", "api", a["archive_download_url"], ">", "artifact.zip"],
            shell=True, check=True
        )
        with zipfile.ZipFile("artifact.zip") as zf:
            zf.extractall(out_dir)
        os.remove("artifact.zip")
        print(f"extracted {a['name']} to {out_dir}")

def check_python_deps():
    scenario_name = os.environ.get("SCENARIO_NAME", "unknown")
    mod_path = pathlib.Path("bench/run_suite.py").resolve()
    spec = importlib.util.spec_from_file_location("run_suite_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import numpy, pandas, lifelines, sksurv, xgboost
    print(f"python deps ok (scenario={scenario_name})")

def format_results():
    from datetime import datetime, timezone

    def fmt_num(v, digits=4):
        if v is None:
            return "—"
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return "—"

    def fmt_status(row):
        status = str(row.get("status", "unknown"))
        if status == "ok":
            return "ok"
        return f"failed: {row.get('error', 'unknown error')}"

    root = pathlib.Path("bench/artifacts")
    results = []
    for p in root.glob("bench_*.json"):
        try:
            results.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"Failed to load {p}: {e}")

    results.sort(key=lambda r: r.get("scenario", ""))
    
    with open("bench/results.nightly.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {len(results)} merged results to bench/results.nightly.json")

    run_url = f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    lines = [
        "## Benchmark Summary",
        f"Run: {run_url}",
        f"Generated: {timestamp}",
        "",
        "| Scenario | Status | N | P | K | Time (s) | Mem (MB) | Epochs | MSE |",
        "|----------|--------|---|---|---|----------|----------|--------|-----|",
    ]
    
    for r in results:
        scen = r.get("scenario", "unknown")
        stat = fmt_status(r)
        
        meta = r.get("metadata", {})
        n_val = meta.get("n", "—")
        p_val = meta.get("p", "—")
        k_val = meta.get("k", "—")
        
        metrics = r.get("metrics", {})
        time_val = fmt_num(metrics.get("time_seconds"))
        mem_val = fmt_num(metrics.get("memory_peak_mb"), digits=1)
        epochs = metrics.get("epochs", "—")
        mse_val = fmt_num(metrics.get("mse_train"))
        
        lines.append(f"| {scen} | {stat} | {n_val} | {p_val} | {k_val} | {time_val} | {mem_val} | {epochs} | {mse_val} |")
    
    summary = "\n".join(lines)
    print(summary)
    
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(summary + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: task_runner.py <task> [args]")
    
    task = sys.argv[1]
    if task == "validate_schemas":
        validate_schemas()
    elif task == "validate_geo_subpop":
        validate_geo_subpop()
    elif task == "build_matrix":
        build_matrix()
    elif task == "extract_maturin_wheel":
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "gamfit"
        extract_maturin_wheel(out_dir)
    elif task == "download_artifacts":
        download_artifacts(sys.argv[2], sys.argv[3])
    elif task == "check_python_deps":
        check_python_deps()
    elif task == "format_results":
        format_results()
    else:
        sys.exit(f"Unknown task: {task}")
