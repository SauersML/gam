from __future__ import annotations

import json

import numpy as np
import pytest

import gamfit.parameter_decomposition as facade


class _RustStub:
    def __init__(self) -> None:
        self.args = None

    def parameter_decomposition_run(self, *args):
        self.args = args
        return (
            '{"schema": "gam.mpd-report", "schema_version": 1}',
            {"clusters/0/basis": np.eye(2)},
        )


REQUEST = {
    "schema": "gam.mpd-request",
    "schema_version": 1,
    "operation": {"kind": "recover_plane_rotations", "tensor": "w", "declared_error": 0.0},
}


def test_facade_marshals_request_and_arrays_without_math(monkeypatch):
    rust = _RustStub()
    monkeypatch.setattr(facade, "rust_module", lambda: rust)
    w = np.arange(9, dtype=np.float32).reshape(3, 3).T
    assert not w.flags.c_contiguous

    out = facade.run_parameter_decomposition(REQUEST, {"w": w})

    assert rust.args is not None
    assert len(rust.args) == 2
    assert json.loads(rust.args[0]) == REQUEST
    sent = rust.args[1]["w"]
    assert sent.dtype == np.float64 and sent.flags.c_contiguous
    assert np.array_equal(sent, w)
    assert set(rust.args[1]) == {"w"}
    assert out.report == {"schema": "gam.mpd-report", "schema_version": 1}
    assert set(out.arrays) == {"clusters/0/basis"}
    assert np.array_equal(out.arrays["clusters/0/basis"], np.eye(2))


def test_facade_never_sends_a_non_finite_request(monkeypatch):
    rust = _RustStub()
    monkeypatch.setattr(facade, "rust_module", lambda: rust)
    request = dict(REQUEST, operation={"kind": "recover_plane_rotations", "tensor": float("nan")})
    with pytest.raises(ValueError):
        facade.run_parameter_decomposition(request, {"w": np.eye(2)})
    assert rust.args is None
    # Positive control: the same stub is reached by a finite request.
    facade.run_parameter_decomposition(REQUEST, {"w": np.eye(2)})
    assert rust.args is not None
