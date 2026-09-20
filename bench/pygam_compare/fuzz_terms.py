"""Seeded term-structure cases for the convergence fuzz plans (``fuzz_terms*``).

A fuzz design is named ``fz<case>``. The *case* number alone fixes the model's
term structure: which term kinds appear (tensor products with two or three
margins, ``ti`` interactions, factor and numeric ``by=`` smooths, fixed
factors with rare levels, random intercepts with 5-2000 levels, cyclic, 2-D
isotropic and shape-constrained smooths, concurvity), their levels, bases and
options. The rep seed then draws the data for that structure at the cell's
``n`` and family, so one case is a fixed formula measured on fresh data.

Everything is a pure function of ``(case, n, family, seed)`` through numpy's
seeded ``default_rng``, so a failing record is reproduced exactly by

    python bench/pygam_compare/worker.py gamfit FAMILY N fz<case> SEED

Triage a run with ``python -m pygam_compare.fuzz_terms RUN_DIR [...]``: every
rep that raised, hung, did not certify or predicted a non-finite value is
tabulated by cause, term kind, family and n. A later input overrides an
earlier record of the same rep, so a ``--designs`` re-run folds into its run.

Held-out rows reuse the training levels of every *fixed* categorical column
(a level unseen in training is a documented schema mismatch for a fixed
factor, not a convergence event); ``group()`` columns keep unseen levels,
which a random effect is specified to tolerate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# The structural stream is keyed apart from the data stream so that a case's
# formula never depends on n, family or seed.
STRUCTURE_KEY = 0xF022
DATA_KEY = 0xDA7A

KINDS: tuple[str, ...] = (
    "te2",
    "te3",
    "ti",
    "by_factor",
    "by_numeric",
    "factor",
    "group",
    "cyclic",
    "iso2d",
    "shape",
    "concurvity",
)
SHAPES: tuple[str, ...] = (
    "monotone_increasing",
    "monotone_decreasing",
    "convex",
    "concave",
)


def is_fuzz_design(design: str) -> bool:
    return design.startswith("fz") and design[2:].isdigit()


def fuzz_design(case: int) -> str:
    return f"fz{case:04d}"


@dataclass
class TermSpec:
    kind: str
    index: int
    params: dict[str, Any] = field(default_factory=dict)

    def col(self, stem: str) -> str:
        return f"{stem}{self.index}"


def _pick(rng: np.random.Generator, options: tuple[Any, ...]) -> Any:
    return options[int(rng.integers(len(options)))]


def case_terms(case: int) -> list[TermSpec]:
    """The term structure of case ``case`` (independent of n/family/seed)."""
    rng = np.random.default_rng([STRUCTURE_KEY, case])
    count = int(rng.choice([1, 2, 3], p=[0.35, 0.4, 0.25]))
    kinds = rng.choice(len(KINDS), size=count, replace=False)
    terms: list[TermSpec] = []
    for index, kind_id in enumerate(sorted(int(k) for k in kinds)):
        kind = KINDS[kind_id]
        params: dict[str, Any] = {}
        if kind in ("by_factor", "factor"):
            params["levels"] = int(rng.integers(2, 31))
            # Small Dirichlet concentration -> a few dominant levels and a
            # tail of rare ones.
            params["alpha"] = float(rng.choice([0.3, 1.0, 5.0]))
            params["singleton"] = bool(rng.random() < 0.4)
            params["empty"] = bool(rng.random() < 0.3)
        if kind == "group":
            params["levels"] = int(np.round(np.exp(rng.uniform(np.log(5), np.log(2000)))))
            params["alpha"] = float(rng.choice([0.5, 5.0]))
        if kind == "te2":
            params["k"] = _pick(rng, (None, 4, 6))
            params["periodic"] = bool(rng.random() < 0.2)
        if kind == "te3":
            params["k"] = _pick(rng, (None, 4))
        if kind == "shape":
            params["shape"] = str(rng.choice(SHAPES))
        if kind == "concurvity":
            # Correlation of the second covariate with the first, and whether
            # the second enters as its own smooth or through a tensor.
            params["noise"] = float(rng.choice([0.02, 0.1, 0.3]))
            params["form"] = str(rng.choice(["additive", "tensor"]))
        if kind == "by_numeric":
            params["centered_z"] = bool(rng.random() < 0.5)
        terms.append(TermSpec(kind, index, params))
    return terms


def case_formula(case: int) -> str:
    return "y ~ " + " + ".join(_term_formula(t) for t in case_terms(case))


def _term_formula(t: TermSpec) -> str:
    c = t.col
    p = t.params
    if t.kind == "te2":
        opts = ""
        if p["k"] is not None:
            opts += f", k={p['k']}"
        if p["periodic"]:
            opts += ", periods=[1, None], origins=[0, None]"
        return f"te({c('ta')}, {c('tb')}{opts})"
    if t.kind == "te3":
        opts = "" if p["k"] is None else f", k={p['k']}"
        return f"te({c('ua')}, {c('ub')}, {c('uc')}{opts})"
    if t.kind == "ti":
        return f"s({c('ia')}) + s({c('ib')}) + ti({c('ia')}, {c('ib')})"
    if t.kind == "by_factor":
        return f"{c('bg')} + s({c('bx')}, by={c('bg')})"
    if t.kind == "by_numeric":
        return f"s({c('nx')}, by={c('nz')})"
    if t.kind == "factor":
        return f"factor({c('fg')})"
    if t.kind == "group":
        return f"group({c('gg')})"
    if t.kind == "cyclic":
        return f"cyclic({c('ct')}, period_start=0, period_end=1)"
    if t.kind == "iso2d":
        return f"s({c('da')}, {c('db')})"
    if t.kind == "shape":
        return f"s({c('sx')}, shape={p['shape']})"
    if t.kind == "concurvity":
        if p["form"] == "additive":
            return f"s({c('ka')}) + s({c('kb')})"
        return f"s({c('ka')}) + te({c('ka')}, {c('kb')})"
    raise ValueError(f"unknown term kind {t.kind!r}")


def _level_codes(
    rng: np.random.Generator,
    rows: int,
    train_rows: int,
    levels: int,
    alpha: float,
    singleton: bool,
) -> NDArray[np.int64]:
    probs = rng.dirichlet(np.full(levels, alpha))
    codes = rng.choice(levels, size=rows, p=probs)
    if singleton and levels >= 2 and rows >= 2:
        # Force the last level to hold exactly one training row. The draw is
        # folded into the training half (the first ``train_rows`` rows), so the
        # level is never held out only; folding rather than redrawing keeps the
        # generator stream, so every other case draws the same data as before.
        codes[codes == levels - 1] = 0
        codes[int(rng.integers(rows)) % train_rows] = levels - 1
    return codes


@dataclass
class FuzzData:
    formula: str
    train: dict[str, Any]
    test: dict[str, Any]
    y: FloatArray
    mu_test: FloatArray
    categorical: dict[str, list[str]]  # column -> declared categories (incl. empty)


def _signal(
    t: TermSpec,
    rng: np.random.Generator,
    rows: int,
    train_rows: int,
    cols: dict[str, Any],
) -> FloatArray:
    c = t.col
    p = t.params
    u: Callable[[], FloatArray] = lambda: rng.uniform(0.0, 1.0, rows)
    if t.kind == "te2":
        a, b = u(), u()
        cols[c("ta")], cols[c("tb")] = a, b
        return np.sin(2 * np.pi * a) * np.cos(2 * np.pi * b)
    if t.kind == "te3":
        a, b, d = u(), u(), u()
        cols[c("ua")], cols[c("ub")], cols[c("uc")] = a, b, d
        return np.sin(2 * np.pi * a) * (b - 0.5) + (d - 0.5) ** 2
    if t.kind == "ti":
        a, b = u(), u()
        cols[c("ia")], cols[c("ib")] = a, b
        return np.sin(2 * np.pi * a) + (b - 0.5) ** 2 + 2 * (a - 0.5) * (b - 0.5)
    if t.kind == "by_factor":
        x = u()
        codes = _level_codes(
            rng, rows, train_rows, p["levels"], p["alpha"], p["singleton"]
        )
        cols[c("bx")] = x
        cols[c("bg")] = codes
        phase = rng.uniform(0.0, 2 * np.pi, p["levels"])
        shift = rng.normal(0.0, 0.5, p["levels"])
        return np.sin(2 * np.pi * x + phase[codes]) * 0.7 + shift[codes]
    if t.kind == "by_numeric":
        x = u()
        z = rng.normal(0.0 if p["centered_z"] else 1.0, 1.0, rows)
        cols[c("nx")], cols[c("nz")] = x, z
        return z * np.sin(2 * np.pi * x) * 0.5
    if t.kind == "factor":
        codes = _level_codes(
            rng, rows, train_rows, p["levels"], p["alpha"], p["singleton"]
        )
        cols[c("fg")] = codes
        return rng.normal(0.0, 0.5, p["levels"])[codes]
    if t.kind == "group":
        codes = _level_codes(rng, rows, train_rows, p["levels"], p["alpha"], False)
        cols[c("gg")] = codes
        return rng.normal(0.0, 0.4, p["levels"])[codes]
    if t.kind == "cyclic":
        x = u()
        cols[c("ct")] = x
        return np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)
    if t.kind == "iso2d":
        a, b = u(), u()
        cols[c("da")], cols[c("db")] = a, b
        return np.exp(-((a - 0.5) ** 2 + (b - 0.4) ** 2) / 0.08)
    if t.kind == "shape":
        x = u()
        cols[c("sx")] = x
        return {
            "monotone_increasing": np.log1p(5 * x),
            "monotone_decreasing": -np.log1p(5 * x),
            "convex": 3 * (x - 0.35) ** 2,
            "concave": -3 * (x - 0.35) ** 2,
        }[p["shape"]]
    if t.kind == "concurvity":
        a = u()
        b = a + rng.normal(0.0, p["noise"], rows)
        cols[c("ka")], cols[c("kb")] = a, b
        return np.sin(2 * np.pi * a)
    raise ValueError(f"unknown term kind {t.kind!r}")


def draw(case: int, n: int, family: str, seed: int) -> FuzzData:
    """Training table/response and held-out table/true mean for one rep."""
    terms = case_terms(case)
    rng = np.random.default_rng([DATA_KEY, case, n, seed])
    rows = 2 * n
    cols: dict[str, Any] = {}
    eta = np.zeros(rows)
    for t in terms:
        eta += _signal(t, rng, rows, n, cols)
    eta = eta - eta.mean()
    scale = np.std(eta)
    if scale > 0:
        eta = eta / scale
    if family == "gaussian":
        mu = eta
        yall = eta + rng.normal(0.0, 0.5, rows)
    elif family == "binomial":
        mu = 1.0 / (1.0 + np.exp(-1.5 * eta))
        yall = (rng.uniform(size=rows) < mu).astype(float)
    elif family == "poisson":
        mu = np.exp(0.5 + 0.7 * eta)
        yall = rng.poisson(mu).astype(float)
    else:
        raise ValueError(f"unknown family {family!r}")

    train: dict[str, Any] = {k: v[:n] for k, v in cols.items()}
    test: dict[str, Any] = {k: v[n:].copy() for k, v in cols.items()}
    categorical: dict[str, list[str]] = {}
    for t in terms:
        for stem in ("bg", "fg", "gg"):
            name = t.col(stem)
            if name not in cols:
                continue
            levels = t.params["levels"]
            declared = [f"L{j}" for j in range(levels)]
            if stem != "gg":
                # A fixed factor predicts only levels it was trained on.
                seen = np.unique(train[name])
                unseen = ~np.isin(test[name], seen)
                test[name][unseen] = seen[0]
                if t.params.get("empty"):
                    # One declared category that no row carries.
                    declared.append(f"L{levels}")
            categorical[name] = declared
            train[name] = np.array([f"L{v}" for v in train[name]], dtype=object)
            test[name] = np.array([f"L{v}" for v in test[name]], dtype=object)
    train["y"] = yall[:n]
    return FuzzData(
        formula=case_formula(case),
        train=train,
        test=test,
        y=yall[:n],
        mu_test=mu[n:],
        categorical=categorical,
    )


def as_frame(table: dict[str, Any], categorical: dict[str, list[str]]) -> Any:
    """The table itself, or — when some factor column declares a category no
    row carries, which a plain label column cannot express — a pandas frame
    whose factor columns are ``Categorical`` over their declared levels."""
    if not any(len(v) > len(set(table[k])) for k, v in categorical.items()):
        return table
    import pandas as pd

    frame = pd.DataFrame(table)
    for name, declared in categorical.items():
        frame[name] = pd.Categorical(frame[name], categories=declared)
    return frame


# ---------------------------------------------------------------------------
# Triage: classify each fuzz record by failure cause and tabulate.
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _message_head(trace: str, head: str | None = None) -> str:
    """The exception line of a recorded traceback with numbers masked, so
    reps that fail the same way at different values group together. ``head``
    is the exception's first line as the worker recorded it; without it (older
    records) the line is recovered from the traceback tail."""
    lines = [ln for ln in trace.strip().splitlines() if ln.strip()]
    # The engine's typed errors end with ``variant:`` / ``category:`` lines;
    # the exception line itself is the first line after the frames.
    variant = next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("variant:")), "")
    if head is None:
        frames_end = max((i for i, ln in enumerate(lines) if ln.startswith("  ")), default=-1)
        head = lines[frames_end + 1] if frames_end + 1 < len(lines) else ""
        head = head.split(": ", 1)[1] if ": " in head else head
    text = f"[{variant}] {head}" if variant else head
    return _NUMBER.sub("#", text)[:110]


def failure_cause(rec: dict[str, Any]) -> str | None:
    """``None`` for a clean rep, otherwise a short cause label.

    Clean means: the rep finished, every phase ran, the summary reports a
    certified optimum, and every point and interval prediction is finite.
    """
    status = rec.get("status")
    if status in ("timeout", "memcap", "crash") or str(status).startswith("not_run"):
        return str(status)
    errors = rec.get("errors") or {}
    if errors:
        phase = next(p for p in ("import", "fit", "pred", "interval", "info") if p in errors)
        kind = (rec.get("error_types") or {}).get(phase, "Exception")
        head = (rec.get("error_heads") or {}).get(phase)
        return f"{phase}:{kind}: {_message_head(errors[phase], head)}"
    if rec.get("certified") is not True:
        return "not_certified"
    if rec.get("pred_finite") is not True:
        return "nonfinite_pred"
    if rec.get("interval_finite") is not True:
        return "nonfinite_interval"
    return None


def triage(records: list[dict[str, Any]]) -> str:
    """Markdown tables: failure rate by cause, by term kind, by family and n."""
    fuzz = [r for r in records if is_fuzz_design(str(r.get("design", "")))]
    total = len(fuzz)
    causes: dict[str, list[dict[str, Any]]] = {}
    for rec in fuzz:
        cause = failure_cause(rec)
        if cause is not None:
            causes.setdefault(cause, []).append(rec)
    failed = sum(len(v) for v in causes.values())
    out = [
        f"fuzz fits: {total}, failed: {failed} ({100.0 * failed / max(total, 1):.1f}%)",
        "",
        "| cause | fits | example (worker.py gamfit FAMILY N DESIGN SEED) |",
        "|---|---|---|",
    ]
    for cause, recs in sorted(causes.items(), key=lambda kv: -len(kv[1])):
        r = recs[0]
        example = f"{r['family']} {r['n']} {r['design']} {r['seed']}"
        out.append(f"| `{cause.replace('|', '/')}` | {len(recs)} | {example} |")

    def rate_table(title: str, key: Callable[[dict[str, Any]], list[str]]) -> None:
        seen: dict[str, list[int]] = {}
        for rec in fuzz:
            bad = int(failure_cause(rec) is not None)
            for k in key(rec):
                cell = seen.setdefault(k, [0, 0])
                cell[0] += bad
                cell[1] += 1
        out.extend(["", f"| {title} | failed / fits |", "|---|---|"])
        for k in sorted(seen):
            bad, n = seen[k]
            out.append(f"| {k} | {bad} / {n} ({100.0 * bad / n:.1f}%) |")

    def kinds_of(rec: dict[str, Any]) -> list[str]:
        return sorted({t.kind for t in case_terms(int(str(rec["design"])[2:]))})

    rate_table("term kind", kinds_of)
    rate_table("family", lambda r: [str(r["family"])])
    rate_table("n", lambda r: [f"{int(r['n']):>5d}"])
    return "\n".join(out) + "\n"


def _load(paths: list[str]) -> list[dict[str, Any]]:
    """Records of every input; a rep recorded again in a later input (a
    ``--designs`` re-run) replaces the earlier record of the same rep."""
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            path = path / "records.jsonl"
        for ln in path.read_text().splitlines():
            if ln:
                rec = json.loads(ln)
                key = (rec["lib"], rec["family"], rec["n"], rec["design"], rec["seed"])
                records[key] = rec
    return list(records.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Triage fuzz_terms records by failure cause.")
    ap.add_argument(
        "inputs",
        nargs="+",
        help="run directories or records.jsonl files; a later input's record "
        "of a rep replaces an earlier one",
    )
    ap.add_argument("--out", type=Path, help="write the markdown here instead of stdout")
    args = ap.parse_args(argv)
    text = triage(_load(args.inputs))
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
