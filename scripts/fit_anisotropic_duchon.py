#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAM_BIN = REPO_ROOT / "target" / "release" / "gam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a multidimensional anisotropic Duchon model with large-data-"
            "friendly defaults. If --z-column is provided, the script builds a "
            "Bernoulli marginal-slope fit and can optionally add main-formula "
            "link deviation and logslope score-warp via linkwiggle(...)."
        )
    )
    parser.add_argument("csv", type=Path, help="Training CSV")
    parser.add_argument("--response", required=True, help="Response column name")
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Continuous feature columns for the main Duchon term",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output model JSON path")
    parser.add_argument("--predict-csv", type=Path, help="Optional CSV to score after fitting")
    parser.add_argument("--predict-out", type=Path, help="Optional prediction CSV output path")
    parser.add_argument("--gam-bin", type=Path, default=DEFAULT_GAM_BIN)
    parser.add_argument(
        "--centers",
        type=int,
        help="Basis center count; defaults to 24 for >=6D, 32 for 4-5D, else 50",
    )
    parser.add_argument("--order", type=int)
    parser.add_argument("--power", type=int)
    parser.add_argument("--length-scale", type=float)
    parser.add_argument(
        "--pure-duchon",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable the wrapper's large-data hybrid default. When not set, "
            "fits with >=6 features default to length_scale=1.0 because the "
            "hybrid Duchon path is much faster and more stable."
        ),
    )
    parser.add_argument(
        "--pilot-subsample-threshold",
        type=int,
        default=2_000,
        help="Forwarded to gam fit; lower default is faster on large high-D fits",
    )
    parser.add_argument(
        "--double-penalty",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adaptive-regularization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--z-column",
        help="If set, build a Bernoulli marginal-slope fit using this latent N(0,1) score column",
    )
    parser.add_argument(
        "--logslope-features",
        nargs="+",
        help="Optional continuous columns for the logslope Duchon term; defaults to --features",
    )
    parser.add_argument(
        "--main-linkwiggle-knots",
        type=int,
        help="Optional main-formula linkwiggle(internal_knots=K) for marginal-slope link deviation",
    )
    parser.add_argument(
        "--score-warp-knots",
        type=int,
        help="Optional logslope linkwiggle(internal_knots=K) for marginal-slope score warp",
    )
    return parser.parse_args()


def dedup_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def default_centers(num_features: int) -> int:
    if num_features >= 6:
        return 24
    if num_features >= 4:
        return 32
    return 50


def build_duchon_term(
    features: list[str],
    *,
    centers: int,
    order: int,
    power: int,
    length_scale: float | None,
    double_penalty: bool,
) -> str:
    if len(features) < 2:
        raise SystemExit("--features needs at least 2 columns for a multidimensional Duchon term")
    parts = [
        ", ".join(features),
        f"centers={centers}",
        f"order={order}",
        f"power={power}",
        f"double_penalty={'true' if double_penalty else 'false'}",
    ]
    if length_scale is not None:
        parts.append(f"length_scale={length_scale}")
    return f"duchon({', '.join(parts)})"


def maybe_add_linkwiggle(terms: list[str], knots: int | None) -> None:
    if knots is not None:
        terms.append(f"linkwiggle(internal_knots={knots})")


def build_formulas(args: argparse.Namespace) -> tuple[str, str | None]:
    features = dedup_columns(args.features)
    logslope_features = dedup_columns(args.logslope_features or features)
    centers = args.centers if args.centers is not None else default_centers(len(features))
    length_scale = resolved_length_scale(args, len(features))
    order = resolved_order(args, len(features), length_scale)
    power = resolved_power(args, len(features), length_scale)

    main_terms = [
        build_duchon_term(
            features,
            centers=centers,
            order=order,
            power=power,
            length_scale=length_scale,
            double_penalty=args.double_penalty,
        )
    ]
    maybe_add_linkwiggle(main_terms, args.main_linkwiggle_knots)
    main_formula = f"{args.response} ~ {' + '.join(main_terms)}"

    if args.z_column is None:
        return main_formula, None

    logslope_terms = [
        build_duchon_term(
            logslope_features,
            centers=centers,
            order=order,
            power=power,
            length_scale=length_scale,
            double_penalty=args.double_penalty,
        )
    ]
    maybe_add_linkwiggle(logslope_terms, args.score_warp_knots)
    return main_formula, " + ".join(logslope_terms)


def resolved_length_scale(args: argparse.Namespace, num_features: int) -> float | None:
    if args.pure_duchon:
        return None
    if args.length_scale is not None:
        return args.length_scale
    if num_features >= 6:
        return 1.0
    return None


def resolved_order(args: argparse.Namespace, num_features: int, length_scale: float | None) -> int:
    if args.order is not None:
        return args.order
    if length_scale is not None and num_features >= 2:
        return 1
    return 0


def resolved_power(args: argparse.Namespace, num_features: int, length_scale: float | None) -> int:
    if args.power is not None:
        return args.power
    if length_scale is not None and num_features >= 2:
        return max(1, num_features // 2)
    return 1


def run_checked(cmd: list[str]) -> None:
    proc: subprocess.Popen[str] | None = None
    handled_signals = (signal.SIGINT, signal.SIGTERM)
    old_handlers = {sig: signal.getsignal(sig) for sig in handled_signals}

    def forward_and_exit(signum: int, _frame: object) -> None:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    try:
        for sig in handled_signals:
            signal.signal(sig, forward_and_exit)
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, start_new_session=True)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    except BaseException:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        raise
    finally:
        for sig, old in old_handlers.items():
            signal.signal(sig, old)


def main() -> None:
    args = parse_args()
    if (args.predict_csv is None) != (args.predict_out is None):
        raise SystemExit("--predict-csv and --predict-out must be provided together")

    main_formula, logslope_formula = build_formulas(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.gam_bin),
        "fit",
        "--adaptive-regularization",
        "true" if args.adaptive_regularization else "false",
        "--scale-dimensions",
        "--pilot-subsample-threshold",
        str(args.pilot_subsample_threshold),
    ]
    if logslope_formula is not None:
        cmd.extend(["--logslope-formula", logslope_formula, "--z-column", args.z_column])
    cmd.extend(["--out", str(args.out), str(args.csv), main_formula])

    print(main_formula, flush=True)
    if logslope_formula is not None:
        print(f"--logslope-formula {logslope_formula}", flush=True)
    run_checked(cmd)

    if args.predict_csv is not None:
        args.predict_out.parent.mkdir(parents=True, exist_ok=True)
        predict_cmd = [
            str(args.gam_bin),
            "predict",
            str(args.out),
            str(args.predict_csv),
            "--out",
            str(args.predict_out),
        ]
        run_checked(predict_cmd)


if __name__ == "__main__":
    main()
