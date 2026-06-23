"""Modal launcher: build + run gam's GPU benches on a real A100.

CUDA is not a Cargo feature in gam — `cudarc` links with `fallback-dynamic-loading`,
so the crate BUILDS with no GPU/CUDA SDK and dlopen's CUDA 12 libs at runtime. The
nvidia/cuda:12.8 devel base image provides cublas/cusolver/cusparse/nvrtc; Modal's
GPU runtime injects libcuda.so. A Volume caches CARGO_HOME + target across runs so
the (large) crate only does a full build once.

Usage:
  modal run gam_gpu_modal.py                      # default: throughput_1412 smoke
  modal run gam_gpu_modal.py --examples "throughput_1412,device_resident_inner_1017"
  modal run gam_gpu_modal.py::build_only          # just build, no run
"""

import modal

CUDA_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .apt_install("git", "curl", "build-essential", "pkg-config", "ca-certificates")
    .run_commands(
        # rustup; the repo's rust-toolchain.toml pins 1.93.0 and auto-installs on first cargo use
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.93.0 --profile minimal",
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "CARGO_HOME": "/cache/cargo",
            "CARGO_TARGET_DIR": "/cache/target",
            # cudarc dlopen path: Modal's injected driver (nvidia dirs) + REAL cuda
            # compute libs + the host driver soname dir. NO stubs/ — the stub libcuda
            # has no device and makes device_count==0 (disables GPU).
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu",
            # CUDA 12 lazy module loading can make cublasCreate/cusolverDnCreate fail
            # NOT_INITIALIZED on some driver/lib combos; force eager loading.
            "CUDA_MODULE_LOADING": "EAGER",
            # writable JIT/compute cache (HOME may be unwritable in the container)
            "CUDA_CACHE_PATH": "/cache/.nv_cache",
        }
    )
)

app = modal.App("gam-gpu")
cache = modal.Volume.from_name("gam-gpu-cache", create_if_missing=True)

REPO = "https://github.com/SauersML/gam.git"
WORKDIR = "/cache/gam"

DEFAULT_EXAMPLES = "throughput_1412"


def _prep_repo(branch: str) -> None:
    import os
    import subprocess

    import shutil

    os.makedirs("/cache/cargo", exist_ok=True)
    # A previously-cached SHALLOW clone trips the build.rs guard — nuke it so we re-clone full.
    if os.path.isfile(WORKDIR + "/.git/shallow"):
        print("[gam-gpu] removing stale shallow clone", flush=True)
        shutil.rmtree(WORKDIR)
    # FULL clone (no --depth): build.rs has a self-integrity guard that runs
    # `git log -1 --format=%an -- build.rs`; a shallow clone has no parent to diff
    # against and misattributes build.rs to the claude[bot] tip commit, tripping it.
    if not os.path.isdir(WORKDIR + "/.git"):
        subprocess.run(["git", "clone", REPO, WORKDIR], check=True)
    # `branch` may be a branch name OR a pinned 40/short hex commit SHA. Pinning to a
    # fixed commit means HEAD never changes across injects even as main moves -> the
    # build cache stays warm and no recompile happens (a no-op reset still bumps source
    # mtimes -> full ~13min rebuild, so we skip checkout entirely when already on target).
    is_sha = len(branch) >= 7 and all(c in "0123456789abcdef" for c in branch.lower())
    # Fetch the exact branch/SHA (a SHA reachable from any advertised ref — incl.
    # refs/verify/* — is fetchable by name on GitHub).
    subprocess.run(["git", "-C", WORKDIR, "fetch", "origin", branch], check=False)
    subprocess.run(["git", "-C", WORKDIR, "fetch", "origin"], check=False)
    head = subprocess.run(["git", "-C", WORKDIR, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if is_sha:
        tgt = subprocess.run(["git", "-C", WORKDIR, "rev-parse", branch], capture_output=True, text=True).stdout.strip()
    else:
        tgt = subprocess.run(["git", "-C", WORKDIR, "rev-parse", "FETCH_HEAD"], capture_output=True, text=True).stdout.strip()
    if head != tgt and tgt:
        subprocess.run(["git", "-C", WORKDIR, "checkout", "-f", branch], check=True)
        if not is_sha:
            subprocess.run(["git", "-C", WORKDIR, "reset", "--hard", f"origin/{branch}"], check=True)
    else:
        print(f"[gam-gpu] already at {head[:12]}; skipping checkout (preserve build cache)", flush=True)
    # Remove stale injected experiment examples (untracked) so build.rs's ban-scanner
    # doesn't trip on leftover probe files; the current run writes its example AFTER this.
    subprocess.run(["git", "-C", WORKDIR, "clean", "-fd", "--", "examples/"], check=False)
    head = subprocess.run(["git", "-C", WORKDIR, "rev-parse", "HEAD"], capture_output=True, text=True)
    print(f"[gam-gpu] repo HEAD = {head.stdout.strip()}", flush=True)


def _sh(cmd: list[str], cwd: str = WORKDIR, **kw):
    import subprocess
    import time

    print(f"\n[gam-gpu] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, **kw)
    print(f"[gam-gpu] (exit {r.returncode}, {time.time()-t0:.1f}s)", flush=True)
    return r


@app.function(image=image, gpu="T4", volumes={"/cache": cache}, timeout=1800)
def run_benches(examples: str = DEFAULT_EXAMPLES, branch: str = "main", build_only: bool = False):
    import subprocess

    _sh(["nvidia-smi"], cwd="/")
    _prep_repo(branch)

    ex_list = [e.strip() for e in examples.split(",") if e.strip()]

    # Build all requested examples first (incremental thanks to the cached target dir).
    for ex in ex_list:
        r = _sh(["cargo", "build", "--release", "--example", ex])
        if r.returncode != 0:
            print(f"[gam-gpu] BUILD FAILED for {ex}", flush=True)
            cache.commit()
            return {"status": "build_failed", "example": ex}
    cache.commit()  # persist target/ so a rerun is incremental even if a run later fails

    if build_only:
        return {"status": "built", "examples": ex_list}

    results = {}
    for ex in ex_list:
        r = _sh(["cargo", "run", "--release", "--example", ex], capture_output=True, text=True)
        out = (r.stdout or "") + "\n----STDERR----\n" + (r.stderr or "")
        print(f"\n========== OUTPUT: {ex} (exit {r.returncode}) ==========\n{out}", flush=True)
        results[ex] = {"exit": r.returncode, "output": out}
    cache.commit()
    return {"status": "ran", "results": {k: v["exit"] for k, v in results.items()}}


@app.function(image=image, gpu="T4", volumes={"/cache": cache}, timeout=1800)
def run_tests(test_target: str = "sae", filters: str = "", branch: str = "main"):
    """Run cargo integration tests on the A100 (the #1026 synthetic K-ladder / EV-vs-K path).

    filters: comma-separated test-name filters; each is run with --exact.
    """
    _sh(["nvidia-smi"], cwd="/")
    _prep_repo(branch)

    flist = [f.strip() for f in filters.split(",") if f.strip()]
    # Build the test binary once.
    r = _sh(["cargo", "test", "--release", "--no-run", "--test", test_target])
    if r.returncode != 0:
        cache.commit()
        return {"status": "build_failed"}
    cache.commit()

    results = {}
    targets = flist or [""]
    for f in targets:
        cmd = ["cargo", "test", "--release", "--test", test_target]
        if f:
            cmd += [f, "--exact"]
        cmd += ["--", "--nocapture", "--test-threads=1"]
        r = _sh(cmd, capture_output=True, text=True)
        out = (r.stdout or "") + "\n----STDERR----\n" + (r.stderr or "")
        print(f"\n========== TEST {f or '(all)'} (exit {r.returncode}) ==========\n{out}", flush=True)
        results[f or "(all)"] = r.returncode
    cache.commit()
    return {"status": "ran", "results": results}


@app.local_entrypoint()
def main(examples: str = DEFAULT_EXAMPLES, branch: str = "main", build_only: bool = False):
    res = run_benches.remote(examples=examples, branch=branch, build_only=build_only)
    print("\n[gam-gpu] RESULT:", res)


@app.function(image=image, gpu="T4", volumes={"/cache": cache}, timeout=1800)
def run_injected(example_name: str, source: str, branch: str = "main"):
    """Inject a local example source into the cloned repo, build, and run it on the A100.
    Lets us iterate on discriminator/fix benches without pushing experimental code to main."""
    import subprocess

    _sh(["nvidia-smi"], cwd="/")
    _prep_repo(branch)
    with open(f"{WORKDIR}/examples/{example_name}.rs", "w") as f:
        f.write(source)
    r = _sh(["cargo", "build", "--release", "--example", example_name])
    if r.returncode != 0:
        cache.commit()
        return {"status": "build_failed"}
    cache.commit()
    r = _sh(["cargo", "run", "--release", "--example", example_name], capture_output=True, text=True)
    out = (r.stdout or "") + "\n----STDERR----\n" + (r.stderr or "")
    print(f"\n========== INJECTED {example_name} (exit {r.returncode}) ==========\n{out}", flush=True)
    return {"status": "ran", "exit": r.returncode}


@app.local_entrypoint()
def tests(filters: str = "", test_target: str = "sae", branch: str = "main"):
    res = run_tests.remote(test_target=test_target, filters=filters, branch=branch)
    print("\n[gam-gpu] TEST RESULT:", res)


@app.local_entrypoint()
def inject(path: str, branch: str = "main"):
    import os

    name = os.path.splitext(os.path.basename(path))[0]
    with open(path) as f:
        source = f.read()
    res = run_injected.remote(example_name=name, source=source, branch=branch)
    print("\n[gam-gpu] INJECT RESULT:", res)


@app.local_entrypoint()
def bench_a100(examples: str = DEFAULT_EXAMPLES, branch: str = "main"):
    # A100 reserved ONLY for the final FP64 perf gates; everything else runs on T4.
    res = run_benches.with_options(gpu="A100-80GB").remote(examples=examples, branch=branch)
    print("\n[gam-gpu] A100 RESULT:", res)


@app.function(image=image, gpu="T4", volumes={"/cache": cache}, timeout=1800)
def run_injected_test(test_name: str, source: str, branch: str = "main"):
    """Inject a tests/<name>.rs, build + run it on the GPU (verify a regression test
    before landing it on main)."""
    _sh(["nvidia-smi"], cwd="/")
    _prep_repo(branch)
    with open(f"{WORKDIR}/tests/{test_name}.rs", "w") as f:
        f.write(source)
    r = _sh(["cargo", "test", "--release", "--no-run", "--test", test_name])
    if r.returncode != 0:
        cache.commit()
        return {"status": "build_failed"}
    cache.commit()
    r = _sh(["cargo", "test", "--release", "--test", test_name, "--", "--nocapture"],
            capture_output=True, text=True)
    out = (r.stdout or "") + "\n----STDERR----\n" + (r.stderr or "")
    print(f"\n========== TEST {test_name} (exit {r.returncode}) ==========\n{out}", flush=True)
    return {"status": "ran", "exit": r.returncode}


@app.local_entrypoint()
def inject_test(path: str, branch: str = "main"):
    import os
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path) as f:
        source = f.read()
    print("\n[gam-gpu] INJECT_TEST RESULT:", run_injected_test.remote(test_name=name, source=source, branch=branch))
