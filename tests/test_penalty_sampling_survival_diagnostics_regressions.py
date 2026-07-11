import json

import pytest

np = pytest.importorskip("numpy")

from gamfit import fit
from gamfit._binding import rust_module
from gamfit._diagnostics import Diagnostics
from gamfit._penalties import (
    ARDPenalty,
    BlockOrthogonalityPenalty,
    OrderedBetaBernoulliPenalty,
    SoftmaxAssignmentSparsityPenalty,
    TotalVariationPenalty,
)
from gamfit._sampling import PosteriorSamples
from gamfit._survival import SurvivalPrediction


def test_penalty_specs_sampling_survival_and_diagnostics_regressions():
    ffi = rust_module()

    penalties = [
        ARDPenalty(target="t"),
        TotalVariationPenalty(weight=1.25, n_eff=4, difference_op="forward_1d", target="t"),
        BlockOrthogonalityPenalty(groups=[[0], [1]], weight=0.75, n_eff=4, target="t"),
        SoftmaxAssignmentSparsityPenalty(k_atoms=3, temperature=0.9, target="t"),
        OrderedBetaBernoulliPenalty(k_max=3, alpha=1.5, tau=0.8, learnable=True, target="t"),
    ]

    descriptors = [p.to_rust_descriptor() for p in penalties]
    latents = json.dumps({"t": {"n": 4, "d": 2}})
    registry_json = ffi.build_analytic_penalty_registry_json(latents, json.dumps(descriptors))
    parsed = json.loads(registry_json)
    kinds = {entry["kind"] for entry in parsed}
    expected = {
        "ard",
        "total_variation",
        "block_orthogonality",
        "softmax_assignment_sparsity",
        "ordered_beta_bernoulli_assignment_map",
    }
    assert expected.issubset(kinds), "Penalty specs should serialize to Rust and round-trip back with expected kinds."

    with pytest.raises((TypeError, ValueError), match="ARDPenalty"):
        ARDPenalty(target={"not": "a target descriptor"})

    rows = [{"x": float(v), "y": 1.0 + 2.0 * float(v)} for v in np.linspace(-1.0, 1.0, 16)]
    model = fit(rows, "y ~ x")
    cfg = dict(samples=20, warmup=10, chains=1, target_accept=0.8, seed=123)
    py_draws = model.sample(rows, **cfg)
    options_json = ffi.build_sample_payload_json(
        cfg["samples"], cfg["warmup"], cfg["chains"], cfg["target_accept"], cfg["seed"]
    )
    # #1512: ffi.sample_table now takes the normalized STRING-cell table that
    # Model.sample() builds via normalize_table (rows: list[list[str]]), not raw
    # float cells — passing floats raises "TypeError: 'float' object is not an
    # instance of 'str' while processing 'rows'". Build the table exactly as the
    # Model.sample() path does so the direct-FFI comparison is apples-to-apples.
    from gamfit._tables import normalize_table

    headers, table_rows, _ = normalize_table(rows)
    payload = ffi.sample_table(model._model_bytes, headers, table_rows, options_json)
    direct_draws = PosteriorSamples.from_ffi_payload(payload, model_bytes=model._model_bytes)
    assert np.array_equal(py_draws.samples, direct_draws.samples), "Python Model.sample and direct Rust sample_table should return identical NUTS draws for the same seed."

    surv = SurvivalPrediction(
        model_class="survival",
        parameters=np.zeros((1, 2), dtype=float),
        times=np.array([1.0, 2.0], dtype=float),
        survival=np.array([[0.8, 0.5]], dtype=float),
    )
    tail = surv.survival_at(np.array([-2.0, 0.0, np.inf], dtype=float))[0]
    assert tail[0] == pytest.approx(1.0), "Survival at negative time should be exactly 1.0."
    assert tail[1] == pytest.approx(1.0), "Survival at zero time should be exactly 1.0."
    assert tail[2] == pytest.approx(0.0), "Survival at infinite time should be exactly 0.0."

    observed = [0.0, 1.0, 2.0]
    predicted = {"mean": [0.0, 1.5, 1.5], "mean_lower": [-0.1, 1.0, 1.0], "mean_upper": [0.1, 2.0, 2.0]}
    diag = Diagnostics.from_predictions(
        formula="y ~ x",
        response_name="y",
        observed=observed,
        predicted=predicted,
    )
    residuals = np.asarray(diag.residuals, dtype=float)
    expected_residuals = np.asarray(observed, dtype=float) - np.asarray(predicted["mean"], dtype=float)
    assert np.allclose(residuals, expected_residuals), "Residuals should equal observed minus predicted mean."
    assert diag.metrics["n_obs"] == pytest.approx(3.0), "n_obs should equal the number of observations."
    assert diag.metrics["mae"] == pytest.approx(np.mean(np.abs(expected_residuals))), "MAE should equal mean absolute residual."
    assert diag.metrics["rmse"] == pytest.approx(np.sqrt(np.mean(expected_residuals**2))), "RMSE should equal square root of mean squared residual."
    assert diag.metrics["bias"] == pytest.approx(np.mean(expected_residuals)), "Bias should equal mean residual."
    y = np.asarray(observed, dtype=float)
    sst = np.sum((y - y.mean()) ** 2)
    sse = np.sum((y - np.asarray(predicted["mean"], dtype=float)) ** 2)
    assert diag.metrics["r_squared"] == pytest.approx(1.0 - sse / sst), "R-squared should match one minus SSE over SST."
