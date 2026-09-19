"""Render docs/benchmarks.md from the committed measurements in bench/pygam_comparison/.

The page is generated so that every number on it traces to a data file, and so
that losses cannot be dropped by hand: each table prints every measured row,
and the summary is computed from the tables.

    python scripts/gen_benchmarks_doc.py            # rewrite docs/benchmarks.md
    python scripts/gen_benchmarks_doc.py --check    # exit 1 if the page is stale
    python scripts/gen_benchmarks_doc.py --refresh-reference-quality
        # re-snapshot the pyGAM rows of the latest reference-quality CI run, then render
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "bench" / "pygam_comparison"
OUT = ROOT / "docs" / "benchmarks.md"
LIVE_QUALITY = ROOT / "bench" / "gha_results" / "reference-quality"
AUDIT_URL = "https://github.com/SauersML/gam/tree/4a84d11a5b37b705551f758f473eb68fe08bf8b0/bench/pygam_audit"
DATA_URL = "https://github.com/SauersML/gam/tree/main/bench/pygam_comparison"
PYGAM_TEST = re.compile(r"^(families|misc|smooths)::quality_vs_(interpret_ml_)?pygam_")


def read_tsv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def ratio(numerator: str, denominator: str) -> str:
    a, b = number(numerator), number(denominator)
    if a is None or b is None:
        return "-"
    return f"{a / b:.2f}x"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


@dataclass(frozen=True)
class AccuracyRow:
    comparison: str
    case: str
    n: str
    family: str
    metric: str
    gamfit: str
    pygam: str
    rel: str
    verdict: str

    @property
    def kind(self) -> str:
        return "truth_mse" if self.metric == "truth_mse" else "heldout"


def accuracy_rows() -> list[AccuracyRow]:
    rows: list[AccuracyRow] = []
    comparison = None
    for line in (DATA / "accuracy_cv.md").read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^### gamfit vs (\S+)", line)
        if heading:
            comparison = heading.group(1)
            continue
        if comparison is None or not line.startswith("| ") or line.startswith("| case "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        case, n, family, metric, gamfit, pygam, _diff, rel, verdict = cells[:9]
        rows.append(AccuracyRow(comparison, case, n, family, metric, gamfit, pygam, rel, verdict.strip("*")))
    return rows


def tally(rows: list[AccuracyRow]) -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        bucket = counts.setdefault((row.comparison, row.kind), {"WIN": 0, "TIE": 0, "LOSS": 0})
        bucket["LOSS" if row.verdict.startswith("LOSS") else row.verdict] += 1
    return counts


def recorded_tally() -> dict[tuple[str, str], dict[str, int]]:
    recorded = {}
    for line in (DATA / "accuracy_cv.md").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\| vs (\S+) \| (\S+) \| (\d+) \| (\d+) \| (\d+) \| \d+ \|$", line)
        if match:
            recorded[(match.group(1), match.group(2))] = {
                "WIN": int(match.group(3)), "TIE": int(match.group(4)), "LOSS": int(match.group(5)),
            }
    return recorded


def fit_failures() -> list[str]:
    text = (DATA / "accuracy_cv.md").read_text(encoding="utf-8")
    errors = text.split("## Errors", 1)[1]
    failures = []
    for line in errors.splitlines():
        match = re.match(r"^- (\S+) fold (\d+) gamfit: IntegrationError: (.*)$", line)
        if match:
            reason = match.group(3)
            reason = "did not certify a stationary optimum" if "did not certify" in reason.split(";")[0] else reason.split(";")[0]
            failures.append(f"{match.group(1)} fold {match.group(2)}: {reason}")
    return failures


def render() -> str:
    fit = read_tsv("fit_cpu.tsv")
    predict = read_tsv("predict_cpu_ms.tsv")
    rss = read_tsv("peak_rss_mb.tsv")
    tour = read_tsv("tour_wall_s.tsv")
    quality = read_tsv("reference_quality_pygam.tsv")
    run = json.loads((DATA / "reference_quality_run.json").read_text(encoding="utf-8"))
    accuracy = accuracy_rows()
    counts = tally(accuracy)
    if counts != recorded_tally():
        raise SystemExit("accuracy_cv.md: per-case rows disagree with its own tally table")
    failures = fit_failures()

    gaussian_large = [
        number(r["gamfit"]) / number(r["pygam_gridsearch"])
        for r in fit
        if r["family"] == "gaussian" and number(r["n"]) >= 1e4
        and number(r["gamfit"]) is not None and number(r["pygam_gridsearch"]) is not None
    ]
    glm = [
        number(r["gamfit"]) / number(r["pygam_gridsearch"])
        for r in fit
        if r["family"] != "gaussian" and number(r["gamfit"]) is not None and number(r["pygam_gridsearch"]) is not None
    ]
    gaussian_small = [
        number(r["gamfit"]) / number(r["pygam_gridsearch"])
        for r in fit
        if r["family"] == "gaussian" and number(r["n"]) < 1e4
        and number(r["gamfit"]) is not None and number(r["pygam_gridsearch"]) is not None
    ]
    timeouts = sorted({f"{r['family']} n={r['n']} {r['design']}" for r in fit if r["gamfit"] == "timeout"})
    binomial_predict = [
        number(r["gamfit_ms"]) / number(r["pygam_ms"]) for r in predict if r["family"] == "binomial"
    ]
    predict_wins = [f"{r['family']} n={r['n']} {r['design']}" for r in predict
                    if number(r["gamfit_ms"]) < number(r["pygam_ms"])]
    rss_ratio = [number(r["gamfit"]) / number(r["pygam_default"]) for r in rss]
    wiggly = sorted({re.sub(r"_n\d+$", "", row.case) for row in accuracy
                     if row.verdict == "LOSS" and row.case.startswith("g1d_")})
    other_losses = sorted({row.case for row in accuracy if row.verdict == "LOSS" and not row.case.startswith("g1d_")})
    failed_cases = sorted({f.split(" fold ")[0] for f in failures})
    by_example = {r["example"]: r for r in tour}
    timed_out_quality = sorted({r["test"].split("::")[1] for r in quality if r["outcome"] == "TIMEOUT"})
    default_heldout = counts[("pygam_default", "heldout")]
    grid_heldout = counts[("pygam_grid", "heldout")]

    audit = json.loads((DATA / "audit_versions.json").read_text(encoding="utf-8"))
    out = []
    out.append("# Benchmarks against pyGAM\n")
    out.append(
        "<!-- Generated by scripts/gen_benchmarks_doc.py from bench/pygam_comparison/. "
        "Edit the data or the script, then rerun it; tests/test_benchmarks_doc.py fails if this page is stale. -->\n"
    )
    out.append(
        f"This page reports every measured comparison between gamfit and pyGAM {audit['pygam_version']}, "
        "the losses included. The numbers come from two sources:\n\n"
        f"- the September 2026 [pyGAM audit]({AUDIT_URL}), which ran gamfit wheel {audit['gamfit_version']} "
        f"against pyGAM {audit['pygam_version']} on a shared 4-CPU machine, and\n"
        f"- the pyGAM rows of the reference-quality CI suite, run "
        f"[{run['run_number']}]({run['run_url']}) on commit `{run['sha'][:10]}`.\n\n"
        f"The measurements are committed in [`bench/pygam_comparison/`]({DATA_URL}), with their provenance "
        "in its README. They describe the versions that were measured. A fix that landed later shows up "
        "here only after the measurement is re-run.\n"
    )
    out.append(
        "The audit machine was heavily loaded (1-minute load average 33-48 on 4 CPUs), so speed is reported as "
        "single-threaded **CPU time** in a fresh process per measurement. Wall time was 4-6x higher. "
        "The tour timings below are wall time and are indicative only.\n"
    )
    out.append(
        "pyGAM is measured two ways. `pygam_default` is `GAM().fit` with its fixed `lam=0.6`, which is fast "
        "because it never estimates smoothness. `pygam_grid` / `pygam_gridsearch` is `GAM().gridsearch()`, "
        "an 11-point GCV/UBRE grid. gamfit always estimates its smoothing parameters by REML/LAML to a "
        "certified optimum. It has no fixed-λ mode, so it is compared with both.\n"
    )

    out.append("## Summary\n")
    out.append("Where gamfit is ahead:\n")
    out.append(
        f"- **Accuracy against pyGAM's defaults.** Over {sum(default_heldout.values())} held-out comparisons "
        f"(5-fold CV), gamfit wins {default_heldout['WIN']}, ties {default_heldout['TIE']} and loses "
        f"{default_heldout['LOSS']}. Against known truth it wins "
        f"{counts[('pygam_default', 'truth_mse')]['WIN']} of {sum(counts[('pygam_default', 'truth_mse')].values())}.\n"
        f"- **Gaussian fits at n ≥ 10⁴.** gamfit uses {min(gaussian_large):.2f}-{max(gaussian_large):.2f}x the CPU "
        "time of pyGAM's grid search, with equal or better error against the true mean.\n"
        "- **Import time.** `import gamfit` takes 0.3-0.5 s, against 1.2-1.8 s for `import pygam`.\n"
        f"- **Tensor-product fits.** The 1500-row `te()` tour example took {by_example['te']['gamfit_s']} s, "
        f"against {by_example['te']['pygam_s']} s for pyGAM.\n"
        f"- **Large-n Gaussian predict.** gamfit predicts faster on {', '.join(predict_wins)}.\n"
        "- **Variance models.** The `mcycle` location-scale fit (`noise_formula=\"s(times)\"`) gives "
        "heteroscedastic observation intervals. pyGAM has no variance model.\n"
    )
    out.append("Where gamfit is behind:\n")
    out.append(
        f"- **Non-Gaussian fit time.** Binomial and Poisson fits take {min(glm):.1f}-{max(glm):.1f}x the CPU "
        "time of pyGAM's grid search, and far longer than pyGAM's fixed-λ default.\n"
        f"- **Small Gaussian fits.** At n = 10³ gamfit uses {min(gaussian_small):.1f}-{max(gaussian_small):.1f}x "
        "the CPU time of pyGAM's grid search.\n"
        f"- **Many smooths.** Fits with 20 smooths ({', '.join(timeouts)}) did not finish within 400 s.\n"
        f"- **Binomial predict.** {min(binomial_predict):.0f}-{max(binomial_predict):.0f}x pyGAM's CPU time.\n"
        f"- **Memory.** Peak RSS is {min(rss_ratio):.1f}-{max(rss_ratio):.1f}x pyGAM's.\n"
        f"- **Very wiggly 1-D truths.** gamfit loses on the {', '.join(wiggly)} cases. The measured wheel "
        "capped the default `s(x)` basis at 8 interior knots, too few for these functions. The default "
        "basis now grows with the data until its own adequacy test passes (#3078; see "
        "[Choosing k](formulas.md#choosing-k)), so these rows describe the old cap until the audit is "
        "re-run.\n"
        f"- **Other accuracy losses:** {', '.join(other_losses)}. "
        f"Against the tuned grid search, gamfit wins {grid_heldout['WIN']}, ties {grid_heldout['TIE']} and "
        f"loses {grid_heldout['LOSS']} of {sum(grid_heldout.values())} held-out comparisons.\n"
        f"- **Fit failures.** gamfit refused {len(failures)} of the cross-validation fits "
        f"({', '.join(failed_cases)}). It raised an error rather than return an unconverged fit.\n"
        f"- **Latency on some paths.** The 300-row monotone example took {by_example['monotone']['gamfit_s']} s "
        f"(pyGAM {by_example['monotone']['pygam_s']} s). The 10,000-row `Default` credit logistic fit did "
        f"not finish. In CI, `{'`, `'.join(timed_out_quality)}` timed out.\n"
    )

    out.append("## Accuracy (5-fold cross-validation)\n")
    out.append(
        "Each case is fitted on four folds and scored on the fifth. Two scores are reported:\n\n"
        "- `heldout` is held-out deviance (Gaussian, Poisson, Gamma) or log-loss (binomial);\n"
        "- `truth_mse` is the error of the fitted mean against the known truth, for simulated cases only.\n\n"
        "A difference counts as a win or a loss only when it is outside two paired standard errors. "
        "A failed gamfit fit counts as a loss. The per-case tables, with standard errors, are in "
        f"[`accuracy_cv.md`]({DATA_URL.replace('/tree/', '/blob/')}/accuracy_cv.md).\n"
    )
    out.append(table(
        ["against", "score", "gamfit wins", "ties", "gamfit loses"],
        [[comparison, kind, str(c["WIN"]), str(c["TIE"]), str(c["LOSS"])] for (comparison, kind), c in counts.items()],
    ) + "\n")
    out.append("Every loss:\n")
    out.append(table(
        ["against", "case", "n", "family", "score", "gamfit", "pyGAM", "change"],
        [[r.comparison, r.case, r.n, r.family, r.metric, r.gamfit, r.pygam, f"{r.rel}%"]
         for r in accuracy if r.verdict == "LOSS"],
    ) + "\n")
    out.append("Fits gamfit refused (counted as losses above):\n")
    out.append("\n".join(f"- {failure}" for failure in failures) + "\n")

    out.append("## Fit CPU time\n")
    out.append(
        "Additive designs `p1`, `p5` and `p20` fit `p` smooths `s(x_j)` to the truth `Σ sin(2πx_j + j)/√p`. "
        "Design `te` fits `te(x0, x1)` to the truth `sin(2πx0)·cos(2πx1)`. "
        "`gamfit_k20` sets `k=20` per smooth, close to pyGAM's 20 splines. "
        "Times are CPU seconds, median of the stated repetitions. `timeout` means the fit did not "
        "finish within 400 s, and `-` means not measured.\n"
    )
    out.append(table(
        ["family", "n", "design", "gamfit", "gamfit k=20", "pyGAM default", "pyGAM gridsearch",
         "gamfit / gridsearch", "reps"],
        [[r["family"], r["n"], r["design"], r["gamfit"], r["gamfit_k20"], r["pygam_default"],
          r["pygam_gridsearch"], ratio(r["gamfit"], r["pygam_gridsearch"]), r["reps"]] for r in fit],
    ) + "\n")

    out.append("## Predict CPU time\n")
    out.append("Milliseconds to predict the posterior mean for n new rows.\n")
    out.append(table(
        ["family", "n", "design", "gamfit ms", "pyGAM ms", "gamfit / pyGAM"],
        [[r["family"], r["n"], r["design"], r["gamfit_ms"], r["pygam_ms"], ratio(r["gamfit_ms"], r["pygam_ms"])]
         for r in predict],
    ) + "\n")

    out.append("## Peak memory\n")
    out.append("Peak resident set size of the fitting process in MB, polled every 50 ms.\n")
    out.append(table(
        ["family", "n", "design", "gamfit", "pyGAM default", "pyGAM gridsearch", "gamfit / pyGAM default"],
        [[r["family"], r["n"], r["design"], r["gamfit"], r["pygam_default"], r["pygam_gridsearch"],
          ratio(r["gamfit"], r["pygam_default"])] for r in rss],
    ) + "\n")

    out.append("## Real-data tour (wall time)\n")
    out.append(
        "These are pyGAM's tour examples and their gamfit equivalents. The gamfit versions are in the "
        "[tour](tour.md). Times are wall seconds on the loaded machine.\n"
    )
    out.append(table(
        ["example", "rows", "pyGAM", "pyGAM s", "gamfit s", "pyGAM result", "gamfit result", "note"],
        [[r["example"], r["rows"], f"`{r['pygam_call']}`", r["pygam_s"], r["gamfit_s"], r["pygam_metric"],
          r["gamfit_metric"], r["note"]] for r in tour],
    ) + "\n")

    out.append("## Reference-quality CI\n")
    out.append(
        "The reference-quality suite re-runs these comparisons on a schedule. It asserts gamfit against the "
        "known truth or a held-out score, and uses pyGAM as the baseline to match or beat. In the metric "
        "column, gamfit's value comes first. This snapshot is from run "
        f"[{run['run_number']}]({run['run_url']}), completed {run['completed_utc']}.\n"
    )
    out.append(table(
        ["outcome", "test", "seconds", "metrics"],
        [[r["outcome"], f"`{r['test'].split('::', 1)[1]}`", r["dur_s"],
          (r["metric"].strip() or r["reason"].strip() or "-").replace("|", "\\|")] for r in quality],
    ) + "\n")

    out.append("## Reproducing\n")
    out.append(
        f"The audit scripts are in [`bench/pygam_audit`]({AUDIT_URL}): `speed/` for the timing and memory "
        "tables, `accuracy/bench_accuracy.py` for the cross-validation, and `docs/` for the tour. After re-running "
        "them, update the files in `bench/pygam_comparison/` and run `python scripts/gen_benchmarks_doc.py`. "
        "To pick up a newer reference-quality run, use "
        "`python scripts/gen_benchmarks_doc.py --refresh-reference-quality`.\n"
    )
    return "\n".join(out)


def refresh_reference_quality() -> None:
    source = LIVE_QUALITY / "quality_results.tsv"
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = [row for row in reader if PYGAM_TEST.match(row["test"])]
    if not rows:
        raise SystemExit(f"{source}: no pyGAM comparison rows")
    with (DATA / "reference_quality_pygam.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(LIVE_QUALITY / "_run.json", DATA / "reference_quality_run.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 if docs/benchmarks.md is stale")
    mode.add_argument("--refresh-reference-quality", action="store_true",
                      help="re-snapshot the pyGAM rows from bench/gha_results/reference-quality first")
    args = parser.parse_args()
    if args.refresh_reference_quality:
        refresh_reference_quality()
    page = render()
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != page:
            print(f"{OUT.relative_to(ROOT)} is stale; run python scripts/gen_benchmarks_doc.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(page, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
