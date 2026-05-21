from __future__ import annotations

import math

import gamfit


def _table(scale: float = 1.0) -> dict[str, list[object]]:
    groups = ["a", "b", "c"]
    x: list[float] = []
    g: list[str] = []
    y: list[float] = []
    for rep in range(12):
        for pos, level in enumerate(groups):
            centered = float(rep - 5.5)
            effect = [-0.6, 0.15, 0.7][pos] * scale
            x.append(centered)
            g.append(level)
            y.append(1.0 + 0.05 * centered + effect + 0.01 * ((rep + pos) % 3 - 1))
    return {"x": x, "g": g, "y": y}


def test_cross_fit_shared_precision_groups_pool_closed_form() -> None:
    left = gamfit.fit(_table(1.0), "y ~ x + group(g)", family="gaussian")
    right = gamfit.fit(_table(0.35), "y ~ x + group(g)", family="gaussian")

    pooled = gamfit.cross_fit_shared_precision_groups(
        {"left": left, "right": right},
        [gamfit.SharedPrecisionGroup("g", shape=3.0, rate=0.5)],
    )

    update = pooled["g"]
    expected = (
        update["n_fits"] * update["dimension"] + 2.0 * (update["shape"] - 1.0)
    ) / (update["quadratic_sum"] + 2.0 * update["rate"])
    assert math.isclose(update["lambda"], expected, rel_tol=1e-12)
    assert update["n_fits"] == 2
    assert update["dimension"] == 3
    assert len(update["fits"]) == 2
    assert update["quadratic_sum"] == sum(
        item["quadratic_contribution"] for item in update["fits"]
    )


def test_cross_fit_shared_precision_groups_accept_per_model_labels() -> None:
    model = gamfit.fit(_table(1.0), "y ~ x + group(g)", family="gaussian")

    pooled = gamfit.cross_fit_shared_precision_groups(
        {"disease_a": model},
        {
            "publication_level": {
                "shape": 2.0,
                "rate": 1.0,
                "labels": {"disease_a": "g"},
            }
        },
    )

    update = pooled["publication_level"]
    assert update["fits"][0]["label"] == "g"
    assert update["n_fits"] == 1
    assert update["dimension"] == 3
