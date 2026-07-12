from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.bsf_zoo_figures import _atlas_cloud_records, fig_gallery
from bench.manifold_zoo_geometry import (
    GEOMETRY_REVISION,
    ZOO,
    ZOO_ORDER,
    declared_atom_spec,
)


def _write_exact_joint_clouds(directory: Path, *, topology_mode: str = "declared") -> Path:
    rng = np.random.default_rng(20260712)
    bases, dims = declared_atom_spec(list(ZOO_ORDER), len(ZOO_ORDER))
    payload: dict[str, np.ndarray] = {}
    factors = []
    for index, kind in enumerate(ZOO_ORDER):
        truth, theta = ZOO[kind].sampler(rng, 24)
        payload[f"true_{index}"] = truth.astype(np.float64)
        payload[f"rec_{index}"] = truth.astype(np.float64)
        payload[f"theta_{index}"] = theta.astype(np.float64)
        factors.append(
            {
                "factor": index,
                "kind": kind,
                "matched_atom": index,
                "r2": 1.0,
            }
        )
    meta = {
        "schema": "joint-manifold-sae-analytic-clouds-v3",
        "coordinate_space": "native-analytic",
        "coordinate_dtype": "float64",
        "geometry_revision": GEOMETRY_REVISION,
        "featurizer": "ours_rust",
        "joint_fit": True,
        "fit_config": {
            "atoms": len(ZOO_ORDER),
            "top_k": 3,
            "assignment": "topk",
            "topology_mode": topology_mode,
            "declared_bases": bases,
            "declared_dims": dims,
        },
        "data_config": {
            "factors": len(ZOO_ORDER),
            "ambient": 32,
            "l0": 3,
            "kinds": list(ZOO_ORDER),
            "dgp": "toy",
        },
        "matching": "hungarian-exact-one-to-one",
        "n_unique_matched_atoms": len(ZOO_ORDER),
        "factors": factors,
    }
    payload["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    path = directory / "clouds_ours_rust_seed0.npz"
    np.savez_compressed(path, **payload)
    return path


def test_atlas_accepts_only_exact_declared_joint_topologies(tmp_path: Path) -> None:
    _write_exact_joint_clouds(tmp_path)
    records, _ = _atlas_cloud_records(tmp_path)
    assert list(records) == list(ZOO_ORDER)

    wrong = tmp_path / "wrong"
    wrong.mkdir()
    _write_exact_joint_clouds(wrong, topology_mode="search")
    with pytest.raises(ValueError, match="exact declared analytic topologies"):
        _atlas_cloud_records(wrong)


def test_gallery_pads_the_one_dimensional_segment_cloud(tmp_path: Path) -> None:
    _write_exact_joint_clouds(tmp_path)
    output = tmp_path / "gallery.png"
    fig_gallery(tmp_path, output, max_factors=1)
    assert output.stat().st_size > 10_000
