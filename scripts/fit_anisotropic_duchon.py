#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAM_BIN = REPO_ROOT / "target" / "release" / "gam"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a multidimensional Duchon smooth on continuous columns with "
            "per-axis length scaling enabled."
        )
    )
    parser.add_argument("csv", type=Path, help="Training CSV")
    parser.add_argument("--response", required=True, help="Response column name")
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Continuous feature columns for the Duchon term",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output model JSON path")
    parser.add_argument("--predict-csv", type=Path, help="Optional CSV to score after fitting")
    parser.add_argument("--predict-out", type=Path, help="Optional prediction CSV output path")
    parser.add_argument("--gam-bin", type=Path, default=DEFAULT_GAM_BIN)
    parser.add_argument("--family", default="gaussian")
    parser.add_argument("--centers", type=int, default=50)
    parser.add_argument("--order", type=int, default=0)
    parser.add_argument("--power", type=int, default=1)
    parser.add_argument("--length-scale", type=float)
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
    return parser.parse_args()


def build_formula(args: argparse.Namespace) -> str:
    features = list(dict.fromkeys(args.features))
    if len(features) < 2:
        raise SystemExit("--features needs at least 2 columns for a multidimensional Duchon term")

    term_parts = [
        ", ".join(features),
        f"centers={args.centers}",
        f"order={args.order}",
        f"power={args.power}",
        f"double_penalty={'true' if args.double_penalty else 'false'}",
    ]
    if args.length_scale is not None:
        term_parts.append(f"length_scale={args.length_scale}")
    return f"{args.response} ~ duchon({', '.join(term_parts)})"


def main() -> None:
    args = parse_args()
    if (args.predict_csv is None) != (args.predict_out is None):
        raise SystemExit("--predict-csv and --predict-out must be provided together")

    formula = build_formula(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.gam_bin),
        "fit",
        "--family",
        args.family,
        "--adaptive-regularization",
        "true" if args.adaptive_regularization else "false",
        "--scale-dimensions",
        "--out",
        str(args.out),
        str(args.csv),
        formula,
    ]

    print(formula)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

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
        subprocess.run(predict_cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
