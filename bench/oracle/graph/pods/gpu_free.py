"""Wait until the job's GPU is free before a MATS job uses it. The cluster does not isolate GPUs: a process left over
from an ended job can keep most of a GPU's memory while Slurm hands that GPU to the next job, which then runs out of
memory at once. Checks the free memory of the visible GPU every 5 minutes, listing the processes on it, and exits 0
once at least --share of it is free, or 1 after --hours.

  gpu_free.py [--share 0.9] [--hours 3]
"""
import argparse
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--share", type=float, default=0.9)
    ap.add_argument("--hours", type=float, default=3.0)
    a = ap.parse_args()
    import torch

    end = time.time() + a.hours * 3600
    while True:
        free, total = torch.cuda.mem_get_info()
        if free >= a.share * total:
            print(f"gpu_free: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free", flush=True)
            return 0
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout
        print(f"gpu_free: {time.strftime('%H:%M')} only {free / 2**30:.1f} of {total / 2**30:.1f} GiB free; processes on the node's GPUs:\n{apps}", flush=True)
        if time.time() > end:
            return 1
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())
