"""#2502 causal steering along atoms of the UNSUPERVISED flagship dictionary.

The K=32k circle-atom dictionary was learned unsupervisedly on wikitext L16
activations. Labels enter only post hoc: calendar prompt clouds identify WHICH
already-learned atom carries weekday (7-cycle) or month (12-cycle) phase, and
evaluate steering along it. Control arm = the standard TopK SAE (also
unsupervised, same data): its best phase latent's decoder direction added at
matched norm. Records share the E1 schema so e1_plots.py renders them.
"""
import argparse, json, math, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e1 import (  # noqa: E402  (reused harness, provenance: gam#2234 E1)
    WEEKDAYS, TAU, load_model_and_tokenizer, resolve_layers, run_clean,
    run_patched, candidate_token_ids, weekday_token_probabilities,
    target_excluded_kl_model_to_base, log,
)

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")

WEEK_FIT = (
    "Today is {label}. Tomorrow is", "If today is {label}, then tomorrow is",
    "The weekday after {label} is", "On a weekly calendar, {label} is followed by",
    "Yesterday was {label}, so today is", "The day that comes right after {label} is",
    "After {label} comes", "Counting forward from {label}, the next day is",
    "A day later than {label} is", "Following {label} on the calendar is",
)
WEEK_BASE = ("Starting on {label}, the next day is",
             "Calendar note: the day after {label} is")
MONTH_FIT = (
    "This month is {label}. Next month is", "The month after {label} is",
    "In the calendar year, {label} is followed by", "After {label} comes",
    "One month later than {label} is", "The month right after {label} is",
    "Following {label} in the year is", "Counting forward from {label}, the next month is",
)
MONTH_BASE = ("Starting in {label}, the next month is",
              "Calendar note: the month after {label} is")


def capture(model, tok, layer, templates, labels):
    acts, logits, li = [], [], []
    for tmpl in templates:
        for i, lab in enumerate(labels):
            a, lg = run_clean(model, tok, layer, tmpl.format(label=lab))
            acts.append(a.numpy().astype(np.float64))
            logits.append(lg)
            li.append(i)
    return np.stack(acts), logits, np.asarray(li)


def circular_r2(phase_turns, idx, n):
    truth = np.exp(1j * TAU * idx / n)
    chart = np.exp(1j * TAU * phase_turns)
    fwd = abs(np.mean(truth * np.conj(chart)))
    rev = abs(np.mean(truth * chart))
    return max(fwd, rev) ** 2, (1 if fwd >= rev else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--prep", default=os.path.expanduser("~/i2502/prep_L16"))
    ap.add_argument("--fits", default=os.path.expanduser("~/i2502/fits"))
    ap.add_argument("--k", type=int, default=32000)
    ap.add_argument("--cycle", choices=("weekday", "month"), default="weekday")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--dose-fractions", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/i2502/steer_flagship"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    labels, fit_t, base_t = ((WEEKDAYS, WEEK_FIT, WEEK_BASE) if args.cycle == "weekday"
                             else (MONTHS, MONTH_FIT, MONTH_BASE))
    n_cyc = len(labels)
    doses = [float(v) for v in args.dose_fractions.split(",")]
    shifts = list(range(1, n_cyc - 1 + 1)) if n_cyc == 7 else [1, 2, 3, 4, 6, 9]

    import gamfit
    with open(os.path.join(args.fits, f"manifold_k{args.k}.pkl"), "rb") as f:
        sae = gamfit.sae.model_from_dict(pickle.load(f))  # K > P carries the support tag (#2567)
    lift = np.load(f"{args.prep}/lift.npy")
    c0 = np.load(f"{args.prep}/c0.npy")
    scale_path = f"{args.prep}/scale.npy"
    s = float(np.load(scale_path)) if os.path.exists(scale_path) else 1.0

    lm, tok = load_model_and_tokenizer(args.model, "", args.dtype)
    layer = resolve_layers(lm)[args.layer]
    cand = candidate_token_ids(tok, " ")[: n_cyc] if n_cyc == 7 else [
        tok.encode(" " + m, add_special_tokens=False)[0] for m in labels]

    log(f"capturing {args.cycle} clouds")
    Xf, _lf, fi = capture(lm, tok, layer, fit_t, labels)
    Xb_amb, base_logits, bi = capture(lm, tok, layer, base_t, labels)
    to_chart = lambda X: np.ascontiguousarray(((X - c0) @ lift.T) / s)
    Zf, Zb = to_chart(Xf), to_chart(Xb_amb)

    # ---- post-hoc atom identification on the fit cloud ----
    codes = np.asarray(sae.encode(Zf))
    active_counts = (codes != 0.0).sum(0)
    cands = np.flatnonzero(active_counts >= max(8, len(Zf) // 6))
    log(f"{len(cands)} atoms active on ≥ threshold of {len(Zf)} {args.cycle} rows")
    lat = sae.converged_latents(Zf)
    best = (-np.inf, None, 1)
    scores = []
    for k in cands:
        ph = np.asarray(lat["coords"][int(k)], dtype=float)[:, 0]
        act = codes[:, k] != 0.0
        if act.sum() < 8:
            continue
        r2, orient = circular_r2(ph[act], fi[act], n_cyc)
        scores.append((float(r2), int(k)))
        if r2 > best[0]:
            best = (r2, int(k), orient)
    r2_best, atom, orient = best
    scores.sort(reverse=True)
    log(f"{args.cycle} atom = {atom} (circular R2={r2_best:.4f}, orient={orient:+d}); "
        f"runners-up {scores[1:4]}")

    # ---- control: torch TopK SAE best phase latent ----
    w = np.load(os.path.join(args.fits, f"torch_topk_k{args.k}.npz"))
    pre = (Zf - w["b_pre"]) @ w["W_enc"] + w["b_enc"]
    kk = 8
    idxs = np.argpartition(pre, -kk, axis=1)[:, -kk:]
    zc = np.zeros_like(pre)
    np.put_along_axis(zc, idxs, np.maximum(np.take_along_axis(pre, idxs, 1), 0), 1)
    design = np.column_stack([np.ones(len(fi)), np.cos(TAU * fi / n_cyc),
                              np.sin(TAU * fi / n_cyc)])
    best_lat, best_r2 = 0, -np.inf
    for j in np.flatnonzero((zc != 0).any(0)):
        col = zc[:, j]
        coef, *_ = np.linalg.lstsq(design, col, rcond=None)
        resid = col - design @ coef
        tss = ((col - col.mean()) ** 2).sum()
        r2 = 1.0 - resid @ resid / max(tss, 1e-30)
        if r2 > best_r2:
            best_r2, best_lat = r2, int(j)
    flat_dir_chart = w["W_dec"][best_lat]
    flat_dir = flat_dir_chart @ (lift * s)
    flat_dir /= max(np.linalg.norm(flat_dir), 1e-30)
    log(f"torch-SAE control latent {best_lat} (phase R2={best_r2:.4f})")

    # ---- steering ----
    fit_coords = np.asarray(sae.coords[atom], dtype=float)[:, 0]
    latb = sae.converged_latents(Zb)
    t0_all = np.asarray(latb["coords"][atom], dtype=float)[:, 0]
    amp_all = np.asarray(latb["assignments"], dtype=float)[:, atom]
    import torch
    records = []
    for i in range(len(Zb)):
        b = int(bi[i])
        prompt = base_t[i // n_cyc].format(label=labels[b])
        base_probs = weekday_token_probabilities(base_logits[i], cand) if n_cyc == 7 \
            else np.exp([float(x) for x in np.log(np.maximum(
                weekday_token_probabilities(base_logits[i], cand), 1e-300))])
        t0 = np.array([t0_all[i]])
        dist = np.abs((fit_coords - t0[0] + 0.5) % 1.0 - 0.5)
        metric_row = int(np.argmin(dist))
        for shift in shifts:
            tgt = (b + shift + 1) % n_cyc
            for dose in doses:
                dcoord = orient * (shift * dose) / n_cyc
                t_to = t0.copy()
                t_to[0] += dcoord
                plan = sae.steer(atom, metric_row, float(amp_all[i]), t0, t_to)
                delta = (np.asarray(plan["delta"], dtype=np.float64) * s) @ lift
                xa = torch.from_numpy(Xb_amb[i].astype(np.float32))
                for arm, d in (("manifold", delta),
                               ("flat", np.linalg.norm(delta) * flat_dir)):
                    pl = run_patched(lm, tok, layer, prompt,
                                     xa + torch.from_numpy(d.astype(np.float32)))
                    probs = weekday_token_probabilities(pl, cand)
                    records.append(dict(
                        arm=arm, base_day=labels[b], base_day_index=b,
                        target_shift_days=int(shift), target_day=labels[tgt],
                        target_day_index=tgt, dose_fraction=float(dose),
                        delta_norm=float(np.linalg.norm(d)),
                        realized_top_weekday_index=int(np.argmax(probs)),
                        target_token_probability=float(probs[tgt]),
                        base_target_token_probability=float(base_probs[tgt]),
                        target_probability_mass_moved=float(probs[tgt] - base_probs[tgt]),
                        collateral_kl_model_to_base_non_target=float(
                            target_excluded_kl_model_to_base(pl, base_logits[i], cand[tgt])),
                        weekday_token_probabilities=[float(x) for x in probs]))
        log(f"context {i+1}/{len(Zb)} done")

    meta = dict(model=args.model, layer_index=args.layer, cycle=args.cycle,
                k=args.k, atom=int(atom), atom_circular_r2=float(r2_best),
                orientation=int(orient), torch_latent=int(best_lat),
                torch_latent_r2=float(best_r2), fit_ev=float("nan"),
                n_contexts=len(Zb), shifts=shifts, doses=doses)
    with open(os.path.join(args.out_dir, "e1_records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    ends = [r for r in records if r["dose_fraction"] == 1.0]
    summary = {}
    for arm in ("manifold", "flat"):
        rs = [r for r in ends if r["arm"] == arm]
        summary[arm] = dict(
            endpoint_target_accuracy=float(np.mean(
                [r["realized_top_weekday_index"] == r["target_day_index"] for r in rs])),
            mean_endpoint_target_token_probability=float(np.mean(
                [r["target_token_probability"] for r in rs])),
            mean_endpoint_mass_moved=float(np.mean(
                [r["target_probability_mass_moved"] for r in rs])),
            mean_endpoint_collateral=float(np.mean(
                [r["collateral_kl_model_to_base_non_target"] for r in rs])))
    with open(os.path.join(args.out_dir, "e1_summary.json"), "w") as f:
        json.dump({"meta": meta, "summary": summary}, f, indent=2)
    print("STEER_FLAGSHIP DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
