"""gam#4563: sweep count of the multi-scale draw selection, against ``n``.

``y ~ factor(g) + te(x, z)`` gives the tensor term one smoothing scale per
margin, so its ``Model.smooth_significance`` row takes the multi-scale replay in
``crates/gam-models/src/fit_orchestration/drivers/smooth_term_lr.rs``: one
``select_draw`` descent per null draw and one for the observation. The issue's
open question is whether the number of sweeps those descents take grows with
``n``. Flat in ``n`` says the coordinate descent contracts on this box and the
``1/eps`` bound is loose. Growth in ``n`` says the stop has to be replaced.

The count is produced by the engine itself (f3c3c06097): once per replayed term
it logs a debug record ``[#4563 multiscale selection] draws=... scales=...
movable=... selections=... sweeps total=... max=... mean=...``. Engine records
reach Python only through the ``gamfit`` logger and only when that logger's
effective level is DEBUG, synced on every engine call. No environment variable
turns them on, so this script lowers the logger itself and keeps the record
with a handler of its own.

The fixture is the one the issue's n = 500 measurement used (seed 31, five
factor levels, ``sin(2 pi x)`` plus N(0, 0.3^2) noise, ``z`` inert). So the
n = 500 row continues that series (5.1 s to fit, 21.7 s to replay at
``RAYON_NUM_THREADS=8``).

Usage:

    python3 experiments/smooth_significance_te_beside_factor_4563.py [--rows 500 2000 8000] [--seed 31]

One line per ``n``: fit wall, replay wall, and the engine's selection record. A
size that raises prints its exception and the next size still runs. Put the
caller's time cap on the whole process: a replay that does not return is
itself the finding for that size.
"""

from __future__ import annotations

import argparse
import logging
import math
import time

import numpy as np

import gamfit
from gamfit._binding import rust_module

FORMULA = "y ~ factor(g) + te(x, z)"
SELECTION_TAG = "[#4563 multiscale selection]"


class SelectionRecords(logging.Handler):
    """Keeps the engine's multi-scale selection records, in arrival order."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if SELECTION_TAG in message:
            self.messages.append(message)


def fixture(n: int, seed: int) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * math.pi * x) + rng.normal(0.0, 0.3, n)
    g = [f"g{index % 5}" for index in range(n)]
    return {"x": list(x), "z": list(z), "g": g, "y": list(y)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[500, 2000, 8000])
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    records = SelectionRecords()
    logger = logging.getLogger("gamfit")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(records)

    for n in args.rows:
        data = fixture(n, args.seed)
        records.messages.clear()
        try:
            started = time.monotonic()
            model = gamfit.fit(data, FORMULA, family="gaussian")
            fit_seconds = time.monotonic() - started
            started = time.monotonic()
            rows = model.smooth_significance(data)
            significance_seconds = time.monotonic() - started
        except Exception as error:  # one size's refusal must not hide the others
            print(f"[4563] n={n} {type(error).__name__}: {error}", flush=True)
            continue
        # Engine records are queued while a call runs; the next engine call
        # delivers whatever is still queued, so make one before reading them.
        rust_module()
        selections = records.messages or [f"no {SELECTION_TAG} record (route not taken)"]
        print(
            f"[4563] n={n} fit={fit_seconds:.1f}s smooth_significance={significance_seconds:.1f}s",
            flush=True,
        )
        for message in selections:
            print(f"[4563] n={n} {message}", flush=True)
        for row in rows:
            print(
                f"[4563] n={n} row name={row.get('name')} p_value={row.get('p_value')} "
                f"statistic_lr={row.get('statistic_lr')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
