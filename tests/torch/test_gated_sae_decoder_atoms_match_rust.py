from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
pytest.importorskip("torch")


def test_gated_sae_decoder_atoms_match_rust() -> None:
    from gamfit import GatedSAEDecoder
    from gamfit._binding import rust_module

    rng = np.random.default_rng(202)
    w_gate = rng.standard_normal((4, 4))
    w_amp = rng.standard_normal((6, 4))
    x = rng.standard_normal((10, 4))

    py_decoder = GatedSAEDecoder(w_gate=w_gate, w_amp=w_amp)
    py_atoms = py_decoder.decode(x)
    rust_atoms = np.asarray(rust_module().gated_sae_decode(x, w_gate, w_amp))

    np.testing.assert_allclose(
        py_atoms,
        rust_atoms,
        rtol=0.0,
        atol=1e-12,
        err_msg="Expected the torch-side gated SAE decoder dictionary atoms to match the Rust decoder atoms for the same inputs and weights.",
    )
