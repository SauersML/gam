"""The simplest-cases atlas: a visual catalog of minimal manifold-SAE fixtures.

Eight tiles, each the smallest possible instance of one capability. Every tile
attempts a real fit with a hard timeout and renders either the fitted manifold
(green title: EV + wall time) or an honest red FAILED/TIMEOUT annotation. The
atlas doubles as a progress scoreboard: re-render after each solver fix and
watch tiles flip green.

Run (box): python examples/simplest_atlas.py --out /mnt/work/exp/figs/atlas.png
"""

import argparse
import multiprocessing as mp
import time
import traceback

import numpy as np


def _tile_circle(rng):
    t = rng.uniform(0, 2 * np.pi, 250)
    x = np.column_stack([np.cos(t), np.sin(t)]) + 0.05 * rng.standard_normal((250, 2))
    return x, dict(K=1, d_atom=1, atom_topology="circle"), None


def _tile_arc(rng):
    t = rng.uniform(0, 1.5 * np.pi, 220)
    x = np.column_stack([np.cos(t), np.sin(t)]) + 0.05 * rng.standard_normal((220, 2))
    return x, dict(K=1, d_atom=1, atom_topology="euclidean"), None


def _tile_line(rng):
    s = rng.uniform(-1.5, 1.5, 220)
    x = np.column_stack([s, 0.6 * s]) + 0.05 * rng.standard_normal((220, 2))
    return x, dict(K=1, d_atom=1, atom_topology="euclidean"), None


def _tile_parabola(rng):
    u = rng.uniform(-1.3, 1.3, 220)
    x = np.column_stack([u, 0.8 * u * u - 0.5]) + 0.05 * rng.standard_normal((220, 2))
    return x, dict(K=1, d_atom=1, atom_topology="euclidean"), None


def _tile_two_circles(rng):
    n = 160
    t1, t2 = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    c1 = np.column_stack([np.cos(t1) - 1.8, np.sin(t1)])
    c2 = 0.7 * np.column_stack([np.cos(t2) + 2.6, np.sin(t2)])
    x = np.vstack([c1, c2]) + 0.05 * rng.standard_normal((2 * n, 2))
    return x, dict(K=2, d_atom=1, atom_topology="circle"), np.repeat([0, 1], n)


def _tile_circle_line(rng):
    n = 160
    t = rng.uniform(0, 2 * np.pi, n)
    circ = np.column_stack([np.cos(t) - 1.6, np.sin(t)])
    s = rng.uniform(-1.4, 1.4, n)
    line = np.column_stack([0.45 * s + 1.7, 0.9 * s])
    x = np.vstack([circ, line]) + 0.05 * rng.standard_normal((2 * n, 2))
    return x, dict(K=2, d_atom=[1, 1], atom_basis=["periodic", "duchon"]), np.repeat([0, 1], n)


def _tile_three_shapes(rng):
    n = 130
    t = rng.uniform(0, 2 * np.pi, n)
    circ = np.column_stack([np.cos(t) - 2.2, np.sin(t)])
    s = rng.uniform(-1.2, 1.2, n)
    line = np.column_stack([0.4 * s, 0.9 * s - 1.6])
    u = rng.uniform(-1.1, 1.1, n)
    par = np.column_stack([u + 2.2, 0.9 * u * u - 0.4])
    x = np.vstack([circ, line, par]) + 0.05 * rng.standard_normal((3 * n, 2))
    return (
        x,
        dict(K=3, d_atom=[1, 1, 1], atom_basis=["periodic", "duchon", "duchon"]),
        np.repeat([0, 1, 2], n),
    )


def _tile_intersecting(rng):
    n = 160
    t = rng.uniform(0, 2 * np.pi, n)
    circ = np.column_stack([np.cos(t), np.sin(t)])
    s = rng.uniform(-1.8, 1.8, n)
    line = np.column_stack([s, 0.35 * s])  # passes through the circle
    x = np.vstack([circ, line]) + 0.04 * rng.standard_normal((2 * n, 2))
    return x, dict(K=2, d_atom=[1, 1], atom_basis=["periodic", "duchon"]), np.repeat([0, 1], n)


def _tile_tiny_n(rng):
    t = rng.uniform(0, 2 * np.pi, 40)
    x = np.column_stack([np.cos(t), np.sin(t)]) + 0.05 * rng.standard_normal((40, 2))
    return x, dict(K=1, d_atom=1, atom_topology="circle"), None


def _tile_high_noise(rng):
    t = rng.uniform(0, 2 * np.pi, 300)
    x = np.column_stack([np.cos(t), np.sin(t)]) + 0.18 * rng.standard_normal((300, 2))
    return x, dict(K=1, d_atom=1, atom_topology="circle"), None


TILES = [
    ("circle", _tile_circle),
    ("arc (3/4 circle, open)", _tile_arc),
    ("line", _tile_line),
    ("parabola", _tile_parabola),
    ("two circles, K=2", _tile_two_circles),
    ("circle + line, K=2", _tile_circle_line),
    ("circle + line + parabola, K=3", _tile_three_shapes),
    ("line THROUGH circle, K=2", _tile_intersecting),
    ("circle, n=40", _tile_tiny_n),
    ("circle, heavy noise", _tile_high_noise),
]


def _fit_worker(name, seed, queue):
    """Runs one tile fit in a subprocess so the parent can enforce a timeout."""
    import gamfit  # noqa: PLC0415 — import inside the worker process

    rng = np.random.default_rng(seed)
    builder = dict(TILES)[name]
    x, kwargs, truth = builder(rng)
    t0 = time.time()
    try:
        m = gamfit.sae_manifold_fit(x, n_iter=15, random_state=0, **kwargs)
        fitted = np.asarray(m.fitted)
        ev = 1.0 - ((x - fitted) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
        out = dict(
            status="ok",
            ev=float(ev),
            seconds=time.time() - t0,
            x=x,
            fitted=fitted,
            coords=[np.asarray(c) for c in m.coords],
            hard=(np.asarray(m.assignments).argmax(1) if truth is not None else None),
            truth=truth,
        )
    except Exception as exc:  # noqa: BLE001 — the atlas reports failures honestly
        out = dict(
            status="fail",
            seconds=time.time() - t0,
            x=x,
            truth=truth,
            error=f"{type(exc).__name__}: {str(exc)[:160]}",
            trace=traceback.format_exc()[-400:],
        )
    queue.put(out)


def run_tile(name, seed, timeout):
    queue = mp.Queue()
    proc = mp.Process(target=_fit_worker, args=(name, seed, queue))
    proc.start()
    try:
        return queue.get(timeout=timeout)
    except Exception:  # noqa: BLE001 — timeout or worker death
        proc.terminate()
        rng = np.random.default_rng(seed)
        x, _, truth = dict(TILES)[name](rng)
        return dict(status="timeout", seconds=timeout, x=x, truth=truth)
    finally:
        proc.join(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tiles = len(TILES)
    n_cols = 5
    n_rows = (n_tiles + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    for ax in axes.ravel()[n_tiles:]:
        ax.set_visible(False)
    from concurrent.futures import ThreadPoolExecutor

    # Tiles run in their own subprocesses already (hard timeout); a small
    # thread pool just overlaps them: the atlas costs max(tile) not sum(tile).
    # Three at a time keeps peak RSS trivial at these shapes.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {name: pool.submit(run_tile, name, args.seed, args.timeout) for name, _ in TILES}
    results = {name: fut.result() for name, fut in futures.items()}
    for ax, (name, _) in zip(axes.ravel(), TILES):
        print(f"[atlas] {name} ...", flush=True)
        res = results[name]
        x = res["x"]
        if res.get("hard") is not None:
            colors = np.where(res["hard"] == res["hard"][0], "tab:blue", "tab:orange")
            ax.scatter(x[:, 0], x[:, 1], c=colors, s=9, alpha=0.55)
        else:
            ax.scatter(x[:, 0], x[:, 1], s=9, alpha=0.5, color="tab:blue")
        if res["status"] == "ok":
            fitted = res["fitted"]
            groups = (
                [np.where(res["hard"] == k)[0] for k in np.unique(res["hard"])]
                if res.get("hard") is not None
                else [np.arange(len(x))]
            )
            for rows in groups:
                if rows.size < 5:
                    continue
                atom_idx = 0 if res.get("hard") is None else int(np.bincount(res["hard"][rows]).argmax())
                coord = res["coords"][min(atom_idx, len(res["coords"]) - 1)][rows, 0]
                order = rows[np.argsort(coord)]
                pts = fitted[order]
                if np.linalg.norm(pts[0] - pts[-1]) < 0.5:
                    pts = np.vstack([pts, pts[:1]])
                ax.plot(pts[:, 0], pts[:, 1], "r-", lw=2)
            ax.set_title(f"{name}\nEV={res['ev']:.4f}  {res['seconds']:.0f}s", color="darkgreen")
        elif res["status"] == "timeout":
            ax.set_title(f"{name}\nTIMEOUT >{args.timeout:.0f}s", color="darkred")
        else:
            ax.set_title(f"{name}\nFAILED: {res['error'][:60]}", color="darkred", fontsize=8)
        print(f"[atlas] {name} -> {res['status']}", flush=True)
        ax.set_aspect("equal")
    fig.suptitle("Simplest-cases atlas — every tile is a real REML fit (red = open bug, re-rendered per fix)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"ATLAS SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
