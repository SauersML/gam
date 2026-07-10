from __future__ import annotations

import numpy as np
import pytest

import gamfit.manifold_crosscoder as facade


class _RustStub:
    def __init__(self) -> None:
        self.args = None

    def sae_crosscoder_fit(self, *args):
        self.args = args
        return {
            "layout": {
                "anchor_label": args[1],
                "anchor_dim": args[0].shape[1],
                "block_dims": [array.shape[1] for array in args[3]],
                "labels": list(args[2]),
                "log_lambda_block": [0.0] * len(args[3]),
            }
        }


def test_crosscoder_facade_marshals_named_targets_without_math(monkeypatch):
    rust = _RustStub()
    monkeypatch.setattr(facade, "rust_module", lambda: rust)
    anchor = np.arange(12, dtype=np.float32).reshape(4, 3)[:, ::-1]
    late = np.arange(8, dtype=np.float32).reshape(4, 2)
    report = facade.sae_crosscoder_fit(
        anchor,
        [("late", late)],
        anchor_label="early",
        n_atoms=2,
        n_harmonics=1,
        run_outer_rho_search=False,
    )
    assert report["layout"]["labels"] == ["late"]
    assert rust.args is not None
    assert rust.args[0].dtype == np.float64 and rust.args[0].flags.c_contiguous
    assert rust.args[3][0].dtype == np.float64 and rust.args[3][0].flags.c_contiguous
    assert rust.args[1] == "early"
    assert rust.args[2] == ["late"]
    assert rust.args[4:6] == (2, 1)


def test_crosscoder_facade_rejects_malformed_target_pairs_before_ffi(monkeypatch):
    monkeypatch.setattr(facade, "rust_module", lambda: _RustStub())
    anchor = np.zeros((3, 2))
    with pytest.raises(TypeError, match="label, array"):
        facade.sae_crosscoder_fit(anchor, [("late", np.zeros((3, 2)), "extra")])
    with pytest.raises(ValueError, match="must be 2-D"):
        facade.sae_crosscoder_fit(anchor, [("late", np.zeros(3))])
