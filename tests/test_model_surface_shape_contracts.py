from __future__ import annotations

import importlib

pytest = importlib.import_module("pytest")
pytest.importorskip("gamfit._rust")


def test_partial_dependence_term_blocks_variance_share_shapes_match_contract() -> None:
    import numpy as np
    import pandas as pd
    import gamfit

    rng = np.random.default_rng(101)
    x = np.linspace(-1.0, 1.0, 50)
    y = 0.7 + 1.4 * x + rng.normal(0.0, 0.05, size=x.shape[0])
    df = pd.DataFrame({"y": y, "x": x})

    model = gamfit.fit(df, "y ~ x")

    assert hasattr(model, "partial_dependence"), "Model must expose partial_dependence with documented signature"
    pd_out = model.partial_dependence(df)
    assert len(pd_out) == len(df), "partial_dependence output must align row-wise with input data"

    blocks = model.term_blocks
    assert isinstance(blocks, tuple) and len(blocks) > 0, "term_blocks must be a non-empty tuple of term descriptors"
    for block in blocks:
        assert block.start <= block.end, "term_blocks entries must have valid coefficient index ranges"

    assert hasattr(model, "variance_share"), "Model must expose variance_share with documented return shape"
    share = model.variance_share(df)
    assert len(share) == len(blocks), "variance_share must provide one entry per term block"
