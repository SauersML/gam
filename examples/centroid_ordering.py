"""Centroid circular-ordering diagnostic: are candidate centers ordered on a ring?

Companion to ``gamfit.adjudicate_atom_shape``. The production race now owns
discrete cyclic structure through its ``ring_clusters`` class, and its
``circle_wins`` flag compares the combined stacking mass of the smooth-circle
and ring-of-clusters candidates against the combined non-circular mass. This
module is an independent ordering diagnostic: it can corroborate the selected
ring-cluster order, or test whether the centroids of a winning free
``mixture`` class nevertheless exhibit circular order against a matched
Gaussian null.

Procedure (validated in the two-tier census of real Qwen3-8B SAE features;
the same construction historically exposed calendar circles before the
ring-of-clusters density landed, and correctly failed rings masked past ~1x
their radius):

  1. Seeded k-means on the 2-D coordinates, with k = the adjudicator's
     ``ring_clusters_reporting_k`` when diagnosing its circular class, or
     ``mixture_reporting_k`` when diagnosing the free-mixture class (k >= 3
     is required for a meaningful ring test). The corresponding
     ``*_fold_selected_k`` values describe honest outer-fold prediction; the
     reporting order is the all-data fit used for this post-race diagnostic.
  2. Ring statistic on the centroids: coefficient of variation (CV) of
     centroid radii about the centroid mean (low CV = centroids sit on a
     circle), plus angular coverage (max angular gap between sorted
     centroid angles; a big gap = an arc or a clump, not a ring).
  3. Monte-Carlo null: centroid sets drawn from a 2-D Gaussian matched to
     the observed centroid mean/covariance; rank-deficient covariance remains
     exactly rank-deficient (no arbitrary ridge); p = P(null CV <= observed CV).

Caveat carried over from the census: sparse non-negative codes pushed
through per-group 2-D PCA produce ring-like centroid arrangements on
STRUCTURELESS controls at a double-digit rate per run. A positive here is
"consistent with a ring", not proof — always run the byte-identical
pipeline on the full-pipeline per-dimension-shuffle and covariance-exact
Hadamard controls before interpreting rates.

Pure NumPy; analysis-side code in the examples/ (non-production) sense of
SPEC.md. Import it next to this file the same way
``harvest_residual_activations.py`` imports ``residual_shard_io``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["kmeans_centroids", "ring_stats", "ring_mc_pvalue",
           "centroid_circular_ordering"]


def _finite_points(values: np.ndarray, name: str, min_rows: int) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < min_rows:
        raise ValueError(
            f"{name} must have shape (n, 2) with n >= {min_rows}; got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive; got {result}")
    return result


def _seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("seed must be an integer in [0, 2**64 - 1]")
    result = int(value)
    if not 0 <= result < 1 << 64:
        raise ValueError(f"seed must be in [0, 2**64 - 1]; got {result}")
    return result


def _center_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center through a relative chart so a large origin does not erase spread."""
    anchor = points[0].copy()
    relative = points - anchor
    if not np.isfinite(relative).all():
        raise ValueError("point coordinate range overflowed while centering")
    mean_offset = relative.mean(axis=0)
    centered = relative - mean_offset
    location = anchor + mean_offset
    if not np.isfinite(centered).all() or not np.isfinite(location).all():
        raise ValueError("centered point coordinates are non-finite")
    return centered, location


def kmeans_centroids(coords: np.ndarray, k: int, seed: int = 0,
                     iters: int = 100) -> np.ndarray:
    """Certified seeded Lloyd k-means on ``(n, 2)`` coordinates."""
    X = _finite_points(coords, "coords", 1)
    k = _positive_integer(k, "k")
    iters = _positive_integer(iters, "iters")
    seed = _seed(seed)
    if k > X.shape[0]:
        raise ValueError(f"k={k} exceeds the {X.shape[0]} coordinate rows")
    centered, location = _center_points(X)
    coordinate_scale = float(np.max(np.abs(centered)))
    if coordinate_scale == 0.0:
        if k == 1:
            return location.reshape(1, 2)
        raise ValueError(f"k={k} exceeds the number of distinct coordinate rows")
    working = centered / coordinate_scale
    if not np.isfinite(working).all():
        raise ValueError("k-means coordinate normalization overflowed")

    rng = np.random.default_rng(seed)
    # Seed one row, then take the farthest row from the selected set. This is
    # deterministic for a seed and refuses a k larger than the distinct support.
    centers = np.empty((k, 2), dtype=np.float64)
    centers[0] = working[int(rng.integers(working.shape[0]))]
    nearest_d2 = np.square(working - centers[0]).sum(axis=1)
    for index in range(1, k):
        selected = int(np.argmax(nearest_d2))
        if nearest_d2[selected] <= 0.0:
            raise ValueError(f"k={k} exceeds the number of distinct coordinate rows")
        centers[index] = working[selected]
        candidate_d2 = np.square(working - centers[index]).sum(axis=1)
        nearest_d2 = np.minimum(nearest_d2, candidate_d2)

    previous_labels: np.ndarray | None = None
    for _ in range(iters):
        d2 = ((working[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        counts = np.bincount(labels, minlength=k)
        # Empty clusters are repaired as part of the deterministic Lloyd map:
        # move the largest-residual row from a cluster that retains another row.
        for empty in np.flatnonzero(counts == 0):
            donors = counts[labels] > 1
            if not donors.any():
                raise ValueError("k-means cannot identify k nonempty clusters")
            residual = d2[np.arange(X.shape[0]), labels]
            donor_row = int(np.argmax(np.where(donors, residual, -np.inf)))
            donor_cluster = int(labels[donor_row])
            counts[donor_cluster] -= 1
            labels[donor_row] = empty
            counts[empty] += 1
        next_centers = np.vstack(
            [working[labels == cluster].mean(axis=0) for cluster in range(k)]
        )
        if not np.isfinite(next_centers).all():
            raise ValueError("k-means centroid update produced non-finite coordinates")
        if previous_labels is not None and np.array_equal(labels, previous_labels):
            result = next_centers * coordinate_scale + location
            if not np.isfinite(result).all():
                raise ValueError("k-means centroid reconstruction overflowed")
            return result
        centers = next_centers
        previous_labels = labels.copy()
    raise RuntimeError(f"k-means did not converge in {iters} iterations")


def ring_stats(centers: np.ndarray) -> tuple[float, float]:
    """Return ``(radius_cv, max_gap_deg)`` for a centroid set.

    ``radius_cv`` is the coefficient of variation of centroid radii about
    the centroid mean (0 = perfect circle). ``max_gap_deg`` is the largest
    angular gap between sorted centroid angles (360 = degenerate).
    """
    c = _finite_points(centers, "centers", 3)
    c, _ = _center_points(c)
    r = np.linalg.norm(c, axis=1)
    mean_radius = float(r.mean())
    if not np.isfinite(mean_radius) or mean_radius <= 0.0:
        raise ValueError("centroid ring is unidentified at zero radius")
    cv = float(r.std() / mean_radius)
    ang = np.sort(np.arctan2(c[:, 1], c[:, 0]))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    return cv, float(np.degrees(gaps.max()))


def ring_mc_pvalue(centers: np.ndarray, observed_cv: float, n_null: int = 2000,
                   seed: int = 0) -> float:
    """P(null radius-CV <= observed) with null centroid sets drawn from a
    2-D Gaussian matched to the observed centroid mean/covariance."""
    c = _finite_points(centers, "centers", 3)
    if not np.isfinite(observed_cv) or observed_cv < 0.0:
        raise ValueError(f"observed_cv must be finite and non-negative; got {observed_cv}")
    n_null = _positive_integer(n_null, "n_null")
    seed = _seed(seed)
    centered, mean = _center_points(c)
    covariance = centered.T @ centered / (c.shape[0] - 1)
    if not np.isfinite(covariance).all():
        raise ValueError("centroid covariance is non-finite")
    trace = float(np.trace(covariance))
    if not np.isfinite(trace) or trace <= 0.0:
        raise ValueError("centroid covariance must have positive finite trace")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 64.0 * covariance.shape[0] * np.finfo(np.float64).eps * trace
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError(
            "centroid covariance is materially indefinite: "
            f"minimum eigenvalue {float(eigenvalues.min()):.6e}"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    factor = eigenvectors * np.sqrt(eigenvalues)[None, :]
    rng = np.random.default_rng(seed)
    k = c.shape[0]
    hits = 0
    for _ in range(n_null):
        z = rng.standard_normal((k, 2)) @ factor.T + mean
        cv, _ = ring_stats(z)
        if cv <= observed_cv:
            hits += 1
    return (hits + 1) / (n_null + 1)


def centroid_circular_ordering(coords: np.ndarray, k: int, *, seed: int = 0,
                               n_null: int = 2000, p_thresh: float = 0.05,
                               gap_thresh_deg: float = 150.0) -> dict:
    """Full second-tier test on a ``(n, 2)`` coordinate cloud.

    ``k`` should be the all-data reporting order of the class being diagnosed:
    ``ring_clusters_reporting_k`` for the constrained circular class or
    ``mixture_reporting_k`` for the free-mixture class. Returns a dict with
    ``radius_cv``, ``max_gap_deg``, ``mc_p``, the centers, and the combined
    verdict ``ordered_on_circle`` (p below threshold AND angular coverage
    without a large gap).
    """
    coords = _finite_points(coords, "coords", 3)
    k = _positive_integer(k, "k")
    n_null = _positive_integer(n_null, "n_null")
    seed = _seed(seed)
    if k < 3:
        raise ValueError(f"need k >= 3 centroids to test a ring; got k={k}")
    if coords.shape[0] < k:
        raise ValueError(f"need at least k={k} rows; got {coords.shape[0]}")
    if not np.isfinite(p_thresh) or not 0.0 < p_thresh < 1.0:
        raise ValueError(f"p_thresh must lie strictly between zero and one; got {p_thresh}")
    if not np.isfinite(gap_thresh_deg) or not 0.0 < gap_thresh_deg <= 360.0:
        raise ValueError(
            f"gap_thresh_deg must lie in (0, 360]; got {gap_thresh_deg}"
        )
    centers = kmeans_centroids(coords, k, seed=seed)
    cv, gap = ring_stats(centers)
    p = ring_mc_pvalue(centers, cv, n_null=n_null, seed=seed ^ 0xCE17_202)
    return {
        "k": int(k),
        "centers": centers,
        "radius_cv": float(cv),
        "max_gap_deg": float(gap),
        "mc_p": float(p),
        "ordered_on_circle": bool(p < p_thresh and gap < gap_thresh_deg),
        "params": {"seed": seed, "n_null": n_null, "p_thresh": p_thresh,
                   "gap_thresh_deg": gap_thresh_deg},
    }
