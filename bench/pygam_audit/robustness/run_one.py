import json, sys, time, warnings, traceback
import numpy as np
from cases import CASES


def fmt(a):
    a = np.asarray(a, dtype=float).ravel()
    return [float(f"{v:.5g}") for v in a[:8]]


def run_gamfit(c):
    import gamfit
    out = {}
    kw = {}
    if c.get("family"):
        kw["family"] = c["family"]
    if c.get("weights"):
        kw["weights"] = c["weights"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = time.time()
        try:
            m = gamfit.fit(c["data"], c["formula"], **kw)
        except Exception as e:
            out["fit"] = "ERROR"
            out["err"] = f"{type(e).__name__}: {str(e)[:600]}"
            out["t"] = round(time.time() - t, 2)
            return out
        out["t"] = round(time.time() - t, 2)
        out["fit"] = "OK"
        out["warnings"] = [str(x.message)[:200] for x in w if "internal knots" not in str(x.message)][:4]
    try:
        s = m.summary()
        out["family"] = s.family_name
        out["edf"] = s.edf_total
        out["conv"] = s.convergence
        out["lambdas"] = fmt(s.lambdas) if s.lambdas is not None else None
    except Exception as e:
        out["summary_err"] = f"{type(e).__name__}: {str(e)[:300]}"
    try:
        r = m.predict(c["pred"], interval=0.95)
        out["pred"] = fmt(r["posterior_mean"])
        out["se"] = fmt(r["posterior_mean_standard_error"])
    except Exception as e:
        out["pred_err"] = f"{type(e).__name__}: {str(e)[:400]}"
        try:
            r = m.predict(c["pred"])
            out["pred"] = fmt(np.asarray(r).ravel() if not isinstance(r, dict) else r["posterior_mean"])
        except Exception as e2:
            out["pred_err2"] = f"{type(e2).__name__}: {str(e2)[:400]}"
    if c.get("truth") is not None and "pred" in out:
        out["truth"] = fmt(c["truth"])
    return out


def run_pygam(c):
    import pygam
    from pygam import s, f, l, te  # noqa
    cls, terms, X, y, w = c["pyg"]
    out = {}
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        t = time.time()
        try:
            g = getattr(pygam, cls)(eval(terms))
            g.fit(X, y, weights=w)
        except Exception as e:
            out["fit"] = "ERROR"
            out["err"] = f"{type(e).__name__}: {str(e)[:600]}"
            out["t"] = round(time.time() - t, 2)
            return out
        out["t"] = round(time.time() - t, 2)
        out["fit"] = "OK"
        out["warnings"] = [str(x.message)[:200] for x in ws][:4]
    out["edf"] = float(g.statistics_["edof"])
    pX = c.get("pred_X")
    if pX is None:
        cols = list(c["pred"].values())
        pX = np.column_stack(cols) if len(cols) > 1 else np.asarray(cols[0])[:, None]
        if X.shape[1] != pX.shape[1]:
            pX = np.column_stack(list(c["pred"].values()))
    try:
        out["pred"] = fmt(g.predict(pX))
        try:
            ci = g.confidence_intervals(pX, width=0.95)
            out["ci"] = [fmt(ci[:, 0]), fmt(ci[:, 1])]
        except Exception as e:
            out["ci_err"] = f"{type(e).__name__}: {str(e)[:200]}"
    except Exception as e:
        out["pred_err"] = f"{type(e).__name__}: {str(e)[:400]}"
    if c.get("truth") is not None and "pred" in out:
        out["truth"] = fmt(c["truth"])
    return out


if __name__ == "__main__":
    lib, name = sys.argv[1], sys.argv[2]
    c = CASES[name]()
    res = run_gamfit(c) if lib == "gamfit" else run_pygam(c)
    print("RESULT " + json.dumps(res, default=str))
