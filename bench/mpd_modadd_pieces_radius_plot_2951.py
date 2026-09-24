"""#2951 figure: trust radius and pieces kept per example along ``mpd_modadd_support_fit_all_2951.py`` fits.

Reads each fit's ``[mall] level ...`` alternation lines (one per support-fit alternation: kept pieces over all
examples, the trust radius after the iteration, and elapsed seconds) and draws both against wall time, one line
per fit.
"""
from __future__ import annotations

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"\[mall\] level ([0-9.e+-]+): kept (\d+) pieces, KL mean ([0-9.]+); step ([0-9.e+-]+) "
                  r"radius ([0-9.e+-]+).*\((\d+)s\)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", nargs="+", required=True, help="label=log pairs")
    parser.add_argument("--examples", type=int, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for spec in args.fits:
        label, path = spec.split("=", 1)
        rows = [tuple(map(float, m.groups())) for m in LINE.finditer(open(path).read())]
        seconds = [r[5] for r in rows]
        ax[0].semilogy(seconds, [r[4] for r in rows], ".-", ms=3, lw=0.8, label=label)
        ax[1].plot(seconds, [r[1] / args.examples for r in rows], ".-", ms=3, lw=0.8, label=label)
    ax[0].set_xlabel("seconds")
    ax[0].set_ylabel("trust radius after the alternation")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("seconds")
    ax[1].set_ylabel("pieces kept per example")
    ax[1].set_yscale("log")
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
