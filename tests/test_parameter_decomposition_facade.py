from __future__ import annotations

import json

import numpy as np
import pytest

import gamfit._parameter_decomposition as facade


class _RustStub:
    def __init__(self) -> None:
        self.args = None

    def parameter_decomposition_run(self, *args):
        self.args = args
        return (
            '{"schema": "gam.mpd-report", "schema_version": 1}',
            {"chart": np.eye(2)},
        )


REQUEST = {
    "schema": "gam.mpd-request",
    "schema_version": 1,
    "operation": {"kind": "linear_closed_chart", "readouts": ["w"], "transitions": ["w"]},
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
    assert set(out.arrays) == {"chart"}
    assert np.array_equal(out.arrays["chart"], np.eye(2))


def test_facade_never_sends_a_non_finite_request(monkeypatch):
    rust = _RustStub()
    monkeypatch.setattr(facade, "rust_module", lambda: rust)
    request = dict(REQUEST, operation={"kind": "linear_closed_chart", "readouts": [float("nan")], "transitions": ["w"]})
    with pytest.raises(ValueError):
        facade.run_parameter_decomposition(request, {"w": np.eye(2)})
    assert rust.args is None
    # Positive control: the same stub is reached by a finite request.
    facade.run_parameter_decomposition(REQUEST, {"w": np.eye(2)})
    assert rust.args is not None


def test_linear_state_quotient_runs_end_to_end_through_rust():
    """One surface op through the real extension: the report is Rust's; the facade only transports it."""
    pytest.importorskip("gamfit._rust")
    readout = np.array([[1.0, 0.0, 0.0]])
    shear = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
    request = {
        "schema": "gam.mpd-request",
        "schema_version": 1,
        "operation": {
            "kind": "linear_state_quotient",
            "readouts": ["m"],
            "transitions": ["t"],
            "chart": {"kind": "close"},
        },
    }

    out = facade.run_parameter_decomposition(request, {"m": readout, "t": shear})

    result = out.report["result"]
    assert result["kind"] == "linear_state_quotient"
    # The shear moves the second coordinate into the observed first: the closed chart is the plane.
    assert result["rows"] == 2
    assert set(out.arrays) == {"chart", "readout_maps/0", "descended/0"}
    assert out.arrays["chart"].shape == (2, 3)
    assert result["quotient_bounds"][0]["upper"] < 1e-12
    # Positive control for the bound: a declared chart without the second coordinate is not closed.
    request["operation"]["chart"] = {"kind": "declared", "tensor": "q"}
    unclosed = facade.run_parameter_decomposition(
        request, {"m": readout, "t": shear, "q": np.array([[1.0, 0.0, 0.0]])}
    )
    assert unclosed.report["result"]["quotient_bounds"][0]["lower"] > 0.5


def test_linear_closed_chart_is_the_quotient_chart_without_its_bounds():
    """The chart-only op returns the chart ``linear_state_quotient`` measures, and nothing else."""
    pytest.importorskip("gamfit._rust")
    readout = np.array([[1.0, 0.0, 0.0]])
    shear = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
    tensors = {"m": readout, "t": shear}
    chart_only = {"kind": "linear_closed_chart", "readouts": ["m"], "transitions": ["t"]}
    out = facade.run_parameter_decomposition(
        {"schema": "gam.mpd-request", "schema_version": 1, "operation": chart_only}, tensors
    )

    assert out.report["result"] == {"kind": "linear_closed_chart", "chart": "chart", "rows": 2}
    assert set(out.arrays) == {"chart"}
    measuring = {"kind": "linear_state_quotient", "readouts": ["m"], "transitions": ["t"], "chart": {"kind": "close"}}
    measured = facade.run_parameter_decomposition(
        {"schema": "gam.mpd-request", "schema_version": 1, "operation": measuring}, tensors
    )
    assert np.array_equal(out.arrays["chart"], measured.arrays["chart"])
