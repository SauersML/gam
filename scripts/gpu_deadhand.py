"""Persistent dead-hand GPU kill-switch for the gam GPU runs.

A DEPLOYED Modal cron (runs every 10 min on cheap CPU, independent of any local
machine) that FULLY stops any GPU app when it's been running too long OR when the
operator has stopped pinging the heartbeat. Belt-and-suspenders on top of the
per-job timeout=1800 already on every GPU function.

Two triggers (whichever fires first):
  1. AGE: any running `gam-gpu*` app alive > MAX_RUN_S (30 min) is stopped.
  2. DEAD-HAND: if the heartbeat file (refreshed by every GPU launch) is older than
     STALE_S (40 min), ALL running `gam-gpu*` apps are stopped immediately — i.e.
     "nobody has checked on the GPUs in a while -> shut everything off."

Deploy once:  modal deploy gam_gpu_deadhand.py
Then it runs forever on Modal's scheduler. The gam-gpu-deadhand app itself is
never targeted.
"""

import modal

app = modal.App("gam-gpu-deadhand")
image = modal.Image.debian_slim().pip_install("modal>=1.0")
cache = modal.Volume.from_name("gam-gpu-cache", create_if_missing=True)

MAX_RUN_S = 1800   # stop any gam-gpu app running longer than 30 min
STALE_S = 2400     # if no GPU launch pinged the heartbeat in 40 min, kill all
HEARTBEAT = "/cache/deadhand_heartbeat"
STATE = "/cache/deadhand_state.json"


@app.function(
    image=image,
    schedule=modal.Period(minutes=10),
    secrets=[modal.Secret.from_name("gam-deadhand-token")],
    volumes={"/cache": cache},
    timeout=240,
)
def deadhand():
    import json
    import os
    import subprocess
    import time

    now = time.time()

    # Heartbeat: refreshed by every GPU launch (launcher writes this at job start).
    try:
        last_ping = float(open(HEARTBEAT).read().strip())
    except Exception:
        last_ping = now  # first run: treat as fresh so we don't nuke a legit job
    heartbeat_stale = (now - last_ping) > STALE_S

    # List apps via the modal CLI (authenticated by the secret env).
    r = subprocess.run(["modal", "app", "list", "--json"], capture_output=True, text=True)
    running = {}
    try:
        for a in json.loads(r.stdout or "[]"):
            name = str(a.get("Description") or a.get("name") or a.get("description") or "")
            state = str(a.get("State") or a.get("state") or "").lower()
            appid = a.get("App ID") or a.get("app_id") or a.get("App Id") or a.get("id")
            is_live = ("ephemeral" in state) or ("running" in state) or ("deployed" in state)
            if appid and name.startswith("gam-gpu") and "deadhand" not in name and is_live:
                running[appid] = name
    except Exception as e:
        print(f"DEADHAND parse error: {e}; raw head: {(r.stdout or '')[:400]}", flush=True)
        return {"status": "parse_error"}

    seen = {}
    if os.path.exists(STATE):
        try:
            seen = json.load(open(STATE))
        except Exception:
            seen = {}

    new_seen = {}
    stopped = []
    for appid, name in running.items():
        first = seen.get(appid, now)
        new_seen[appid] = first
        too_old = (now - first) > MAX_RUN_S
        if too_old or heartbeat_stale:
            res = subprocess.run(["modal", "app", "stop", appid, "--yes"], capture_output=True, text=True)
            reason = "heartbeat_stale" if heartbeat_stale else "age>30m"
            stopped.append(f"{appid}({name},{reason},rc={res.returncode})")
            new_seen.pop(appid, None)

    json.dump(new_seen, open(STATE, "w"))
    cache.commit()
    msg = (
        f"DEADHAND now={int(now)} heartbeat_age={int(now-last_ping)}s stale={heartbeat_stale} "
        f"tracked={len(running)} stopped={stopped}"
    )
    print(msg, flush=True)
    return {"status": "ok", "stopped": stopped, "tracked": len(running)}
