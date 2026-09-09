#!/usr/bin/env python3
"""Render gh#2234 / gh#2263-item-3's requested-vs-realized displacement figure.

Reads the `[#2234 ...]` lines that
`crates/gam-sae/src/inference/tests_unit_speed_steering_2234.rs` prints under
`--nocapture`, so every number in the figure comes from a test run rather than
from this file. Usage:

    python3 artifacts/issue_2234_unit_speed_steering.py <green.log> [more.log ...]

Writes `artifacts/issue_2234_unit_speed_steering.json` (the parsed measurement)
and `artifacts/issue_2234_unit_speed_steering.svg` (the figure). Dependency-free
on purpose: this has to run wherever the log lands.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(
    r"\[#2234 (?P<fixture>[^/\]]+)/(?P<arm>raw|canonical)\] \+(?P<k>\d+)/12 = "
    r"(?P<requested>[-\d.e+]+) \| (?P<mean>[-\d.e+]+) \| (?P<median>[-\d.e+]+) \| "
    r"(?P<lo>[-\d.e+]+) \| (?P<hi>[-\d.e+]+) \| (?P<worst>[-\d.e+]+)"
)
CHART = re.compile(
    r"\[#2234 (?P<fixture>[^/\]]+)\] EV=(?P<ev>[-\d.e+]+) .*?speed_cv=(?P<cv>[-\d.e+]+), "
    r"speed max/min=(?P<ratio>[-\d.e+]+), fidelity floor=(?P<floor>[-\d.e+]+) arc, "
    r"perimeter=(?P<perimeter>[-\d.e+]+)"
)


def parse(paths):
    measurement = {"charts": {}, "sweeps": {}}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                chart = CHART.search(line)
                if chart:
                    fixture = chart.group("fixture")
                    if fixture.startswith("minor"):
                        measurement["charts"][fixture] = {
                            key: float(chart.group(key))
                            for key in ("ev", "cv", "ratio", "floor", "perimeter")
                        }
                row = ROW.search(line)
                if row:
                    key = f"{row.group('fixture')}/{row.group('arm')}"
                    entry = {
                        key2: float(row.group(key2))
                        for key2 in ("requested", "mean", "median", "lo", "hi", "worst")
                    }
                    entry["k"] = int(row.group("k"))
                    measurement["sweeps"].setdefault(key, []).append(entry)
    for rows in measurement["sweeps"].values():
        rows.sort(key=lambda r: r["k"])
    return measurement


def panel(x0, y0, width, height, fixture, sweeps, chart):
    """One fixture: realized/requested against the requested twelfth."""
    raw = sweeps.get(f"{fixture}/raw", [])
    canonical = sweeps.get(f"{fixture}/canonical", [])
    if not raw:
        return ""
    parts = [f'<g transform="translate({x0},{y0})">']
    # A ratio axis: 1.0 is "the request was landed".
    lo_ratio, hi_ratio = 0.0, 1.8

    def yy(ratio):
        return height - (ratio - lo_ratio) / (hi_ratio - lo_ratio) * height

    def xx(k):
        return (k - 0.5) / 6.0 * width

    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" '
        f'stroke="#c8ccd4"/>'
    )
    for ratio in (0.5, 1.0, 1.5):
        colour = "#111827" if ratio == 1.0 else "#e5e7eb"
        dash = "" if ratio == 1.0 else ' stroke-dasharray="3 3"'
        parts.append(
            f'<line x1="0" y1="{yy(ratio):.1f}" x2="{width}" y2="{yy(ratio):.1f}" '
            f'stroke="{colour}"{dash}/>'
        )
        parts.append(
            f'<text x="-6" y="{yy(ratio) + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="#6b7280">{ratio:.1f}</text>'
        )
    for entry in raw:
        k = entry["k"]
        req = entry["requested"]
        x = xx(k)
        parts.append(
            f'<line x1="{x:.1f}" y1="{yy(entry["lo"] / req):.1f}" x2="{x:.1f}" '
            f'y2="{yy(entry["hi"] / req):.1f}" stroke="#dc2626" stroke-width="9" '
            f'stroke-linecap="round" opacity="0.30"/>'
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{yy(entry["mean"] / req):.1f}" r="3.2" '
            f'fill="#dc2626"/>'
        )
    for entry in canonical:
        k = entry["k"]
        req = entry["requested"]
        x = xx(k) + 9
        parts.append(
            f'<line x1="{x:.1f}" y1="{yy(entry["lo"] / req):.1f}" x2="{x:.1f}" '
            f'y2="{yy(entry["hi"] / req):.1f}" stroke="#2563eb" stroke-width="9" '
            f'stroke-linecap="round" opacity="0.30"/>'
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{yy(entry["mean"] / req):.1f}" r="3.2" '
            f'fill="#2563eb"/>'
        )
    for entry in raw:
        parts.append(
            f'<text x="{xx(entry["k"]) + 4:.1f}" y="{height + 15}" text-anchor="middle" '
            f'font-size="11" fill="#6b7280">+{entry["k"]}/12</text>'
        )
    ratio = chart.get("ratio", float("nan")) if chart else float("nan")
    cv = chart.get("cv", float("nan")) if chart else float("nan")
    parts.append(
        f'<text x="0" y="-22" font-size="13" font-weight="600" fill="#111827">'
        f'{fixture}: chart speed max/min = {ratio:.2f}</text>'
    )
    parts.append(
        f'<text x="0" y="-7" font-size="11" fill="#6b7280">'
        f'speed_cv = {cv:.3e}</text>'
    )
    parts.append("</g>")
    return "".join(parts)


def render(measurement, path):
    fixtures = sorted(measurement["charts"], key=lambda f: -float(f[len("minor") :]))
    pw, ph, gap = 300, 210, 62
    width = 70 + len(fixtures) * (pw + gap)
    height = ph + 175
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#f9fafb"/>',
        '<text x="24" y="30" font-size="17" font-weight="700" fill="#111827">'
        "Requested vs realized on-manifold displacement (gh#2234, gh#2263 item 3)</text>",
        '<text x="24" y="50" font-size="12" fill="#374151">'
        "realized / requested, per row; whisker = min..max over 192 rows, dot = mean. "
        "1.0 means the request was landed.</text>",
    ]
    for index, fixture in enumerate(fixtures):
        svg.append(
            panel(
                70 + index * (pw + gap),
                105,
                pw,
                ph,
                fixture,
                measurement["sweeps"],
                measurement["charts"].get(fixture),
            )
        )
    legend_y = 105 + ph + 48
    svg.append(
        f'<rect x="70" y="{legend_y - 11}" width="14" height="9" fill="#dc2626" '
        f'opacity="0.45"/>'
    )
    svg.append(
        f'<text x="92" y="{legend_y - 2}" font-size="12" fill="#111827">'
        "raw chart step (what steer_rows applies today)</text>"
    )
    svg.append(
        f'<rect x="400" y="{legend_y - 11}" width="14" height="9" fill="#2563eb" '
        f'opacity="0.45"/>'
    )
    svg.append(
        f'<text x="422" y="{legend_y - 2}" font-size="12" fill="#111827">'
        "canonical (arc-length) step — steer_rows_unit_speed</text>"
    )
    svg.append(
        f'<text x="70" y="{legend_y + 20}" font-size="12" fill="#374151">'
        "The red DOT sits on 1.0 in every panel: pooled over rows the raw steer is exactly "
        "right. The red WHISKER is the defect.</text>"
    )
    svg.append("</svg>")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(svg))


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    measurement = parse(sys.argv[1:])
    if not measurement["sweeps"]:
        sys.stderr.write("no [#2234 ...] sweep lines found in the supplied logs\n")
        return 1
    with open(
        os.path.join(HERE, "issue_2234_unit_speed_steering.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(measurement, handle, indent=1, sort_keys=True)
        handle.write("\n")
    render(measurement, os.path.join(HERE, "issue_2234_unit_speed_steering.svg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
