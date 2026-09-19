"""#2502 flagship: overcomplete manifold dictionary on Qwen3.5-4B-Base L16.

Arms (append-per-record JSONL, resumable):
  pilot      — K=512 circle atoms, jumprelu, 20k rows (fast signal)
  manifold   — K=<--k> circle atoms (d_atom=1), jumprelu, full train
  linear     — matched-K TopK linear lane (softmax assignment, same top_k)

Each arm records train EV, held-out EV (frozen-decoder OOS solve), mean L0 and
alive-atom count on test codes, wall, and saves the model artifact (pickle of
to_dict) for the interpretation / steering / splice phases.
"""
import argparse, json, os, pickle, resource, sys, time, traceback
import numpy as np


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def emit(path, rec):
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[fit] {json.dumps(rec)[:400]}", flush=True)


def ev_of(X, R):
    return 1.0 - float(((X - R) ** 2).sum()) / float((X ** 2).sum())


def code_stats(model, X, chunk=2000):
    """Held-out code sparsity + alive atoms. Handles both the dense lane
    (encode -> (N, K) array) and the support-sparse lane (encode -> dict with
    sparse indices/values)."""
    l0s = []
    alive = None
    alive_set = set()
    for i in range(0, len(X), chunk):
        codes = model.encode(np.ascontiguousarray(X[i:i + chunk]))
        if isinstance(codes, dict):
            vals = np.asarray(codes["values"])
            idxs = np.asarray(codes["indices"])
            nz = np.abs(vals) > 0
            l0s.append(nz.sum(1))
            alive_set.update(np.unique(idxs[nz]).tolist())
        else:
            arr = np.asarray(codes)
            nz = arr != 0.0
            l0s.append(nz.sum(1))
            a = nz.any(0)
            alive = a if alive is None else (alive | a)
    l0 = np.concatenate(l0s)
    if alive is None:
        return float(l0.mean()), len(alive_set), sorted(alive_set)
    return float(l0.mean()), int(alive.sum()), alive



def lane_kwargs(lane, top_k):
    """Per-lane validated recipes: support-sparse minimal contract for topk;
    the bench/massive_k_manifold_validate regime for threshold_gate."""
    if lane == "topk":
        return dict(top_k=top_k, sparsity_weight=0.0)
    return dict(sparsity_weight=0.01, smoothness_weight=0.01,
                isometry_weight=0.0, learning_rate=1.0)


def run_arm(out, name, base, fitfn, X_train, X_test):
    t0 = time.perf_counter()
    try:
        model = fitfn()
        wall = time.perf_counter() - t0
        rec = {**base, "record": name, "status": "ok", "wall_s": round(wall, 1),
               "peak_rss_gb": round(peak_rss_gb(), 2)}
        rec["train_ev"] = ev_of(X_train, np.asarray(model.fitted))
        try:
            rec["reconstruction_r2"] = float(model.reconstruction_r2)
        except Exception:
            pass
        t1 = time.perf_counter()
        R = np.asarray(model.reconstruct(np.ascontiguousarray(X_test)))
        rec["test_ev"] = ev_of(X_test, R)
        rec["oos_wall_s"] = round(time.perf_counter() - t1, 1)
        l0, n_alive, alive = code_stats(model, X_test)
        rec["test_mean_l0"] = round(l0, 2)
        rec["alive_atoms_test"] = n_alive
        try:
            rec["description_length_bits"] = model.description_length()
        except Exception as e:  # noqa: BLE001
            rec["description_length_bits"] = f"<err {type(e).__name__}>"
        try:
            ir = model.incoherence_report
            rec["mu_hat"] = None if ir is None else ir.get("mu_hat")
        except Exception:
            pass
        emit(out, rec)
        return model, alive
    except Exception as exc:  # noqa: BLE001
        emit(out, {**base, "record": name, "status": type(exc).__name__,
                   "error": str(exc)[:800], "traceback_tail": traceback.format_exc()[-1500:],
                   "wall_s": round(time.perf_counter() - t0, 1)})
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=os.path.expanduser("~/i2502/prep_L16"))
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--pilot-k", type=int, default=512)
    ap.add_argument("--pilot-rows", type=int, default=20000)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--lane", default="topk",
                    help="assignment lane for the manifold arm (topk | threshold_gate)")
    ap.add_argument("--gpu", default="auto",
                    help="gam gpu policy ("
                         "route-not-refuse fix, so its device lane hard-errors "
                         "on small decoder-smoothness groups)")
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=["pilot", "manifold", "linear"])
    ap.add_argument("--pilot-lanes", nargs="+", default=["topk", "threshold_gate"])
    ap.add_argument("--tag", default="", help="suffix for pilot record names")
    ap.add_argument("--smooth", type=float, default=None, help="initial smoothness seed for the LAML seminorm")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/i2502/fits"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "fits.jsonl")

    import gamfit
    X_train = np.ascontiguousarray(np.load(f"{args.prep}/train.npy"))
    X_test = np.ascontiguousarray(np.load(f"{args.prep}/test.npy"))
    meta = json.load(open(f"{args.prep}/meta.json"))
    print(f"[fit] train {X_train.shape} test {X_test.shape} gamfit "
          f"{gamfit.__version__} prep={meta}", flush=True)
    base = dict(layer=meta["layer"], p=meta["p"], n_train=len(X_train),
                n_test=len(X_test), gamfit=gamfit.__version__)

    if "pilot" in args.arms:
        Xp = X_train[: args.pilot_rows]
        for lane in args.pilot_lanes:
            model, _ = run_arm(
                out, f"pilot_{lane}_k{args.pilot_k}{args.tag}",
                {**base, "k": args.pilot_k, "n_train": len(Xp), "lane": lane},
                lambda lane=lane: gamfit.sae_manifold_fit(
                    Xp, K=args.pilot_k, d_atom=1,
                    assignment=lane, n_iter=args.n_iter, random_state=args.seed,
                    gpu=args.gpu, **lane_kwargs(lane, args.top_k),
                    **({} if args.smooth is None else {"smoothness_weight": args.smooth})),
                Xp, X_test)
            del model

    if "manifold" in args.arms:
        model, alive = run_arm(
            out, f"manifold_k{args.k}", {**base, "k": args.k, "lane": args.lane},
            lambda: gamfit.sae_manifold_fit(
                X_train, K=args.k, d_atom=1,
                assignment=args.lane, n_iter=args.n_iter, random_state=args.seed,
                gpu=args.gpu, **lane_kwargs(args.lane, args.top_k)),
            X_train, X_test)
        if model is not None:
            with open(os.path.join(args.out_dir, f"manifold_k{args.k}.pkl"), "wb") as f:
                pickle.dump(model.to_dict(), f, protocol=4)
            if alive is not None:
                np.save(os.path.join(args.out_dir, f"manifold_k{args.k}_alive.npy"),
                        np.asarray(alive))
            print("[fit] manifold artifact saved", flush=True)
        del model

    if "linear" in args.arms:
        model, _ = run_arm(
            out, f"linear_k{args.k}", {**base, "k": args.k},
            lambda: gamfit.sae_manifold_fit(
                X_train, K=args.k, assignment="softmax", top_k=args.top_k,
                n_iter=30, random_state=args.seed, gpu=args.gpu),
            X_train, X_test)
        if model is not None:
            with open(os.path.join(args.out_dir, f"linear_k{args.k}.pkl"), "wb") as f:
                pickle.dump(model.to_dict(), f, protocol=4)
        del model

    print("[fit] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
