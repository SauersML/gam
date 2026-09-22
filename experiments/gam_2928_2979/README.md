# gam#2928 / gam#2979 workload, through the release CLI

## Why this exists

gam#2979's body names its harness as **`bench_2928.truth.rs` in the lane root, MSI-only**, and
its artefacts as `acl42:/scratch.global/sauer354/swarm17/gam-2928/runs/t2370/`. Both strings are
quoted from the issue. Nobody on this side verified either, and a `find` over
`/scratch.global/sauer354` returns nothing matching `*2928*`, so the harness is gone or was never
at that path.

It is not needed. The workload is an ordinary `gam fit` request, so the **release CLI runs it** —
which is a better receipt than an MSI-only bench binary, because it times the production entry
point rather than a fixture beside it.

## FIDELITY — read this before comparing any number to the issue

gam#2979 states every parameter of the data except one: **where each row's uniform draw comes
from**. The score stream is pinned (`SplitMix(0x2370_2941)`), the index is pinned
(`q(t, x) = −6.99 + 1.706·ln t + 0.25·sex`, slope `0.8`), the ages are pinned
(`20 + i%5`, `40 + i%13`), but the event-time draw is not.

The issue records **691 events at 2,000 rows**, which is the check. Sixteen readings of the
underspecified parts were enumerated — mid-quantile versus SplitMix draw, both signs of the
normal quantile, both signs of the slope, and left truncation counted or not:

| | events at 2,000 rows |
|---|---|
| range over all sixteen | 540 – 831 |
| closest to 691 | 636 |
| this script's default | 553 |

**None reproduces 691.** The target is inside the range but is not hit by any of them, so it is
not a sign convention or an off-by-one — the draw is defined somewhere this side cannot see. The
issue names that place: `gnomon-swarm17/gnomon-2370/truth-module.rs`, "ported unchanged". That
file is the fixture's definition, and recovering it is what makes the numbers below continuous
with the filed ones.

**What this harness still answers without it.** Both arms read the same `data.csv`, so:

- **gam#2928's acceptance is a ratio** between the arms and stays valid. It re-baselines: a
  ratio measured here is not a continuation of the filed 5–10× or the measured 2.36×, and must
  be reported as its own series.
- **gam#2979's question is which phase dominates**, which is a property of the request shape.
  The event fraction differs (27.6% here against the issue's 34.6%), so the fit is easier and
  the absolute walls will be smaller.

Swap in the gnomon module's draw when it is recovered and both series reconnect.

## Files

- `generate_truth2370.py` — writes `data.csv` and the two request documents. It computes
  `survival_time_anchor` (earliest entry) and `baseline_scale` (mean exit age) from the data it
  just wrote, so the requests cannot drift from their dataset. It prints the event count.
- `run_ab.sh` — the interleaved A/B driver.

## Invocations

```bash
# The dataset and its two requests, for one size.
python3 generate_truth2370.py --rows 2000 --out-dir /scratch.global/$USER/gam2928/n2000

# gam#2928: the interleaved acceptance A/B, both sizes, three repeats, one job, one node.
./run_ab.sh target/release/gam /scratch.global/$USER/gam2928 2400 300000

# gam#2979: the same script per binary — unpatched main first, because run A is the phase
# attribution and run B only tells you what moved.
./run_ab.sh /path/to/gam-main   /scratch.global/$USER/gam2979-A 2000
./run_ab.sh /path/to/gam-fixed  /scratch.global/$USER/gam2979-B 2000
```

One fit on its own, for a quick look:

```bash
target/release/gam fit data.csv --request request_anchored.json --out model.json -v
```

`-v` puts the CLI at `Debug`, which is where the `[OUTER]` and `[STAGE]` lines live: the release
build's equivalent of the retired harness's `BENCH_LOG=1`. `-vv` is `Trace`.

`run_ab.sh` prints what to read out of the logs for each issue, including the continuation's
certified depth — the line `#2661 anchored continuation certified at N steps`. `N = 4` is the
ladder's floor and confirms the reading in `fix/2661-2928/`; `8`, `16` or
`RefinementBudgetExhausted` would make the acceptance bar the next thing to price.
