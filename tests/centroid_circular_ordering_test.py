"""Unit tests for examples/centroid_ordering.py (the tier-2 ring test).

Three synthetic cases mirror the injections the construction was validated
on against real-activation censuses:

  * clusters arranged on a ring       -> ordered_on_circle True
  * the same points, per-dim shuffled -> ordered_on_circle False
  * a ring masked by a dominant linear factor, seen through top-2 PCA
    (the census's own projection step) -> ordered_on_circle False
    (this is the documented top-variance projection failure mode)
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from unittest import mock
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MOD_PATH = _REPO_ROOT / "examples" / "centroid_ordering.py"
_SPEC = importlib.util.spec_from_file_location("centroid_ordering", _MOD_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"failed to load centroid_ordering from {_MOD_PATH}")
_CO: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CO
_SPEC.loader.exec_module(_CO)


def _clusters_on_ring(k: int = 7, per: int = 40, noise: float = 0.08,
                      seed: int = 0) -> np.ndarray:
    """k tight Gaussian clusters centered on a unit circle."""
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(k):
        th = 2 * np.pi * i / k
        c = np.array([np.cos(th), np.sin(th)])
        pts.append(c + noise * rng.standard_normal((per, 2)))
    return np.concatenate(pts, axis=0)


def _top2_pca(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


class CentroidCircularOrderingTest(unittest.TestCase):
    def test_ring_of_clusters_passes(self) -> None:
        coords = _clusters_on_ring(seed=0)
        out = _CO.centroid_circular_ordering(coords, 7, seed=0, n_null=2000)
        self.assertTrue(out["ordered_on_circle"], msg=str(out))
        self.assertLess(out["mc_p"], 0.05)
        self.assertLess(out["radius_cv"], 0.2)
        self.assertLess(out["max_gap_deg"], 150.0)
        self.assertEqual(out["params"]["kmeans_seed"], 0)
        self.assertEqual(out["params"]["control_seed"], 0xCE17_202)
        self.assertEqual(
            out["params"]["null_model"],
            "per_dimension_permutation_refit_seeded_lloyd",
        )

    def test_shuffled_ring_fails(self) -> None:
        rng = np.random.default_rng(1)
        coords = _clusters_on_ring(seed=1)
        # independent per-dimension shuffle destroys the ring but preserves
        # the marginals (the census's structureless-control construction)
        shuffled = np.column_stack([rng.permutation(coords[:, 0]),
                                    rng.permutation(coords[:, 1])])
        out = _CO.centroid_circular_ordering(shuffled, 7, seed=1, n_null=2000)
        self.assertFalse(out["ordered_on_circle"], msg=str(out))

    def test_masked_ring_after_pca_fails(self) -> None:
        # ring in dims (0, 1) + a linear factor in dim 2 at 2x the ring
        # radius: top-2 PCA keeps the linear factor and one ring axis, so
        # the ring is lost before the test ever sees it (a top-variance
        # projection failure mode; the adjudicator degrades to mixture too)
        rng = np.random.default_rng(2)
        ring = _clusters_on_ring(seed=2)
        n = ring.shape[0]
        lin = 2.0 * rng.uniform(-1.0, 1.0, size=(n, 1))
        X3 = np.concatenate([ring, lin], axis=1)
        coords = _top2_pca(X3)
        out = _CO.centroid_circular_ordering(coords, 7, seed=2, n_null=2000)
        self.assertFalse(out["ordered_on_circle"], msg=str(out))

    def test_input_validation(self) -> None:
        coords = _clusters_on_ring(seed=3)
        with self.assertRaises(ValueError):
            _CO.centroid_circular_ordering(coords, 2)
        with self.assertRaises(ValueError):
            _CO.centroid_circular_ordering(coords[:, :1], 5)
        with self.assertRaises(ValueError):
            _CO.centroid_circular_ordering(np.full((8, 2), np.nan), 3)
        with self.assertRaises(ValueError):
            _CO.centroid_circular_ordering(coords, 7, n_null=0)
        with self.assertRaises(ValueError):
            _CO.kmeans_centroids(np.zeros((8, 2)), 2)
        with self.assertRaises(TypeError):
            _CO.centroid_circular_ordering(
                np.ones((8, 2), dtype=np.complex128) * (1.0 + 2.0j), 3
            )

    def test_single_cluster_is_the_mean_not_the_seed_row(self) -> None:
        coords = np.array([[0.0, 0.0], [2.0, 4.0], [7.0, -1.0]])
        centers = _CO.kmeans_centroids(coords, 1, seed=17)
        np.testing.assert_allclose(centers[0], coords.mean(axis=0), atol=1.0e-15)

    def test_ring_stats_are_scale_invariant_when_raw_squares_overflow(self) -> None:
        angles = 2.0 * np.pi * np.arange(7) / 7.0
        centers = np.column_stack((np.cos(angles), np.sin(angles)))
        reference_cv, reference_gap = _CO.ring_stats(centers)
        # Directly subtracting two antipodal x coordinates now overflows, but
        # the geometry itself is representable in a bounded affine chart.
        scaled_cv, scaled_gap = _CO.ring_stats(1.7e308 * centers)
        self.assertTrue(np.isfinite(scaled_cv))
        self.assertTrue(np.isfinite(scaled_gap))
        self.assertAlmostEqual(scaled_cv, reference_cv, places=14)
        self.assertAlmostEqual(scaled_gap, reference_gap, places=12)

    def test_conditional_randomization_refits_lloyd_and_preserves_marginals(
        self,
    ) -> None:
        coords = np.column_stack(
            (
                np.arange(12, dtype=np.float64),
                np.array([5, 1, 9, 2, 8, 0, 7, 3, 11, 4, 10, 6], dtype=np.float64),
            )
        )
        pristine = coords.copy()
        base_order = np.arange(coords.shape[0])

        class CertifiedShuffle:
            def __init__(self) -> None:
                self.calls = 0

            def shuffle(self, order: np.ndarray) -> None:
                np.testing.assert_array_equal(order, base_order)
                draw = self.calls // 2
                if self.calls % 2 == 0:
                    order[:] = np.roll(base_order, draw + 1)
                else:
                    order[:] = np.roll(base_order[::-1], draw + 2)
                self.calls += 1

        shuffle = CertifiedShuffle()
        fitted_inputs: list[np.ndarray] = []

        def certify_refit(draw: np.ndarray, k: int, seed: int) -> np.ndarray:
            self.assertEqual(k, 3)
            self.assertEqual(seed, 17)
            if not fitted_inputs:
                np.testing.assert_array_equal(draw, coords)
            else:
                for column in range(coords.shape[1]):
                    np.testing.assert_array_equal(
                        np.sort(draw[:, column]), np.sort(coords[:, column])
                    )
            fitted_inputs.append(draw.copy())
            return np.array([[1.0, 0.0], [-0.5, 0.75], [-0.5, -0.75]])

        with (
            mock.patch.object(
                _CO.np.random, "default_rng", return_value=shuffle
            ) as make_rng,
            mock.patch.object(_CO, "kmeans_centroids", side_effect=certify_refit),
            mock.patch.object(
                _CO,
                "ring_stats",
                side_effect=[
                    (0.20, 120.0),
                    (0.10, 120.0),
                    (0.20, 120.0),
                    (0.40, 120.0),
                ],
            ),
        ):
            p_value = _CO.ring_mc_pvalue(
                coords,
                3,
                n_null=3,
                control_seed=23,
                kmeans_seed=17,
            )

        make_rng.assert_called_once_with(23)
        self.assertEqual(shuffle.calls, 6)
        self.assertEqual(len(fitted_inputs), 4)
        np.testing.assert_array_equal(fitted_inputs[0], coords)
        np.testing.assert_array_equal(fitted_inputs[1][:, 0], coords[np.roll(base_order, 1), 0])
        np.testing.assert_array_equal(
            fitted_inputs[1][:, 1], coords[np.roll(base_order[::-1], 2), 1]
        )
        np.testing.assert_array_equal(coords, pristine)
        self.assertEqual(p_value, 0.75, "ties and the plus-one correction must count")


if __name__ == "__main__":
    unittest.main()
