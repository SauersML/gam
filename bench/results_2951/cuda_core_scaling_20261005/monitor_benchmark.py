"""Record process memory around a Rust benchmark; no model computation here."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
command = sys.argv[2:]
if command and command[0] == "--":
    command = command[1:]
started = time.monotonic()
process = subprocess.Popen(command)
samples = []
errors = 0
while process.poll() is None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        for row in result.stdout.splitlines():
            pid, memory = (part.strip() for part in row.split(",", 1))
            if int(pid) == process.pid:
                samples.append({"seconds": time.monotonic() - started,
                                "gpu_memory_mib": int(memory)})
    except (OSError, ValueError, subprocess.SubprocessError):
        errors += 1
    time.sleep(0.5)
code = process.wait()
report = {
    "command": command, "pid": process.pid, "exit_code": code,
    "wall_seconds": time.monotonic() - started,
    "peak_child_host_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    "sampled_peak_process_gpu_mib": max((s["gpu_memory_mib"] for s in samples), default=None),
    "gpu_samples": samples, "sampling_errors": errors,
    "scope": "Linux host high-water RSS and sampled nvidia-smi process GPU allocation. "
             "GPU sampling can miss short peaks; it is not an allocator memory bound. "
             "Wall time includes model import and benchmark setup. Timing includes monitoring overhead.",
}
(destination / "PROCESS_MEMORY.json").write_text(json.dumps(report, indent=2) + "\n")
sys.exit(code)
