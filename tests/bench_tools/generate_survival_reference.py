#!/usr/bin/env python3
"""Generate trusted survival reference artifacts for integration tests."""

import json
import sys
from pathlib import Path

TRUSTED_REFERENCE_JSON = {
    "age_entry": [
        50.0,
        50.0,
        50.0,
        50.0,
        51.0,
        52.0,
        53.0,
        54.0
    ],
    "age_exit": [
        58.0,
        61.0,
        64.0,
        67.0,
        59.0,
        62.0,
        65.0,
        68.0
    ],
    "artifacts": {
        "age_basis": {
            "degree": 2,
            "knot_vector": {
                "data": [
                    -2.3025850929940455,
                    -2.3025850929940455,
                    -2.3025850929940455,
                    -0.5697527492387704,
                    1.1630795945165047,
                    2.89591193827178,
                    2.89591193827178,
                    2.89591193827178
                ],
                "dim": [
                    8
                ],
                "v": 1
            }
        },
        "age_transform": {
            "delta": 0.1,
            "minimum_age": 50.0
        },
        "calibrator": None,
        "coefficients": {
            "data": [
                25.835898378632805,
                1.7869161146790287,
                29.14999902772817,
                52.37658172620565,
                0.0,
                0.0
            ],
            "dim": [
                6
            ],
            "v": 1
        },
        "companion_models": [],
        "hessian_factor": None,
        "interaction_metadata": [],
        "penalties": [
            {
                "column_range": {
                    "end": 4,
                    "start": 0
                },
                "lambda": 0.5,
                "matrix": {
                    "data": [
                        1.0,
                        -2.0,
                        1.0,
                        0.0,
                        -2.0,
                        5.0,
                        -4.0,
                        1.0,
                        1.0,
                        -4.0,
                        5.0,
                        -2.0,
                        0.0,
                        1.0,
                        -2.0,
                        1.0
                    ],
                    "dim": [
                        4,
                        4
                    ],
                    "v": 1
                },
                "order": 2
            }
        ],
        "reference_constraint": {
            "reference_log_age": 2.537018622388447,
            "transform": {
                "data": [
                    0.0,
                    -0.02979749466534877,
                    -0.4860884205942921,
                    -0.8734014865317182,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.9991121093116685,
                    -0.014484217119546228,
                    -0.026025176135636558,
                    0.0,
                    -0.014484217119546228,
                    0.7637180473641466,
                    -0.4245503491329098,
                    0.0,
                    -0.026025176135636558,
                    -0.4245503491329098,
                    0.2371698433241848
                ],
                "dim": [
                    5,
                    4
                ],
                "v": 1
            }
        },
        "static_covariate_layout": {
            "column_names": [
                "sex"
            ],
            "ranges": [
                {
                    "max": 0.0,
                    "min": 0.0
                },
                {
                    "max": 0.0,
                    "min": 0.0
                }
            ]
        },
        "time_varying_basis": None
    },
    "event_competing": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ],
    "event_target": [
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1
    ],
    "extra_static_covariates": [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    ],
    "extra_static_names": [],
    "pcs": [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
    ],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ],
    "sample_weight": [
        1.0,
        2.0,
        3.0,
        4.0,
        1.0,
        1.0,
        1.0,
        1.0
    ],
    "sex": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ],
    "static_covariates": [
        [
            0.0,
            0.0
        ],
        [
            0.0,
            0.0
        ],
        [
            0.0,
            0.0
        ],
        [
            0.0,
            0.0
        ],
        [
            0.0,
            0.0
        ],
        [
            0.0,
            0.0
        ],
        [
            0.0,
            0.0
        ],
        [
            0.0,
            0.0
        ]
    ],
    "lifelines_cif": [
        0.6321210934815431,
        0.6321207264171644,
        0.6321204109495415,
        0.6321201334574286
    ],
    "lifelines_weighted_brier": 0.286330117856061
}


def main() -> int:
    if len(sys.argv) != 2:
        print('Usage: generate_survival_reference.py <output_path>', file=sys.stderr)
        return 1

    output_path = Path(sys.argv[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as fp:
        json.dump(TRUSTED_REFERENCE_JSON, fp, indent=2)
        fp.write('\n')

    return 0


if __name__ == '__main__':
    sys.exit(main())
