"""gam#3054: every gamfit save writes through the one Rust saved-model writer.

``Model.save`` used ``Path.write_bytes``, which truncates the file at ``path``
before writing it, so a save that failed part way lost the user's previous
model. The save now goes through ``gam_model_api::saved_model::write_saved_model``:
the bytes are written and synced to a temporary sibling, which is renamed over
``path``. A save that fails leaves ``path`` as it was and leaves no temporary
file. A filesystem refusal raises the ``OSError`` subclass its kind names.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gamfit


def _fit() -> gamfit.Model:
    rng = np.random.default_rng(3054)
    x = rng.uniform(0.0, 1.0, size=200)
    y = np.sin(2.0 * np.pi * x) + 0.1 * rng.standard_normal(200)
    return gamfit.fit(pd.DataFrame({"x": x, "y": y}), "y ~ s(x)")


def test_save_replaces_the_file_whole_and_a_failed_save_leaves_it_untouched(tmp_path: Path) -> None:
    model = _fit()
    path = tmp_path / "model.gam"
    path.write_bytes(b"previous")
    model.save(path)
    assert path.read_bytes() == model.dumps()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.gam"]

    # The temporary file is written, then the rename over a non-empty
    # directory fails: nothing at the path changes and nothing is left behind.
    occupied = tmp_path / "occupied.gam"
    occupied.mkdir()
    (occupied / "kept").write_bytes(b"kept")
    with pytest.raises(OSError, match="occupied.gam"):
        model.save(occupied)
    assert (occupied / "kept").read_bytes() == b"kept"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.gam", "occupied.gam"]

    with pytest.raises(FileNotFoundError, match="missing"):
        model.save(tmp_path / "missing" / "model.gam")
    with pytest.raises(FileNotFoundError, match="missing"):
        model.save(str(tmp_path / "missing" / "model.gam"))

    reloaded = gamfit.load(path)
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 7)})
    np.testing.assert_array_equal(
        np.asarray(reloaded.predict(frame)), np.asarray(model.predict(frame))
    )
