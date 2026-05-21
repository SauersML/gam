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


def test_cross_fit_shared_precision_groups_skip_fits_where_group_is_absent() -> None:
    grouped = gamfit.fit(_table(1.0), "y ~ x + group(g)", family="gaussian")
    ungrouped = gamfit.fit(_table(1.0), "y ~ x", family="gaussian")

    pooled = gamfit.cross_fit_shared_precision_groups(
        [grouped, ungrouped],
        [{"name": "g", "shape": 1.5, "rate": 0.25}],
    )

    assert pooled["g"]["n_fits"] == 1
    assert pooled["g"]["fits"][0]["model"] == 0


def test_cross_fit_shared_precision_groups_reject_bad_hyperpriors() -> None:
    model = gamfit.fit(_table(1.0), "y ~ x + group(g)", family="gaussian")

    for group in (
        {"name": "g", "shape": 0.0, "rate": 1.0},
        {"name": "g", "shape": 1.0, "rate": -0.1},
    ):
        try:
            gamfit.cross_fit_shared_precision_groups([model], [group])
        except ValueError as exc:
            assert "shared precision group" in str(exc)
        else:
            raise AssertionError("bad hyperprior should be rejected")


def test_cross_fit_shared_precision_groups_reject_duplicate_names() -> None:
    model = gamfit.fit(_table(1.0), "y ~ x + group(g)", family="gaussian")

    try:
        gamfit.cross_fit_shared_precision_groups(
            [model],
            [
                gamfit.SharedPrecisionGroup("g"),
                gamfit.SharedPrecisionGroup("g"),
            ],
        )
    except ValueError as exc:
        assert "duplicate shared precision group name" in str(exc)
    else:
        raise AssertionError("duplicate group names should be rejected")


def test_cross_fit_shared_precision_groups_reject_no_matches() -> None:
    model = gamfit.fit(_table(1.0), "y ~ x + group(g)", family="gaussian")

    try:
        gamfit.cross_fit_shared_precision_groups([model], [gamfit.SharedPrecisionGroup("missing")])
    except ValueError as exc:
        assert "did not match any model coefficients" in str(exc)
    else:
        raise AssertionError("unmatched shared group should be rejected")


def test_cross_fit_shared_precision_groups_reject_dimension_mismatch() -> None:
    three_levels = gamfit.fit(_table(1.0), "y ~ x + group(g)", family="gaussian")
    full = _table(1.0)
    keep = [idx for idx, level in enumerate(full["g"]) if level != "c"]
    two_level_data = {key: [values[idx] for idx in keep] for key, values in full.items()}
    two_levels = gamfit.fit(two_level_data, "y ~ x + group(g)", family="gaussian")

    try:
        gamfit.cross_fit_shared_precision_groups(
            [three_levels, two_levels],
            [gamfit.SharedPrecisionGroup("g")],
        )
    except ValueError as exc:
        assert "inconsistent dimensions" in str(exc)
    else:
        raise AssertionError("dimension mismatch should be rejected")


def test_cross_fit_shared_precision_groups_do_not_match_provenance_source() -> None:
    model = gamfit.fit(_table(1.0), "y ~ x + group(g)", family="gaussian")

    try:
        gamfit.cross_fit_shared_precision_groups([model], [gamfit.SharedPrecisionGroup("group")])
    except ValueError as exc:
        assert "did not match any model coefficients" in str(exc)
    else:
        raise AssertionError("source='group' must not be treated as a precision label")
