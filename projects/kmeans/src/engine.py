"""Transparent k-means engine that records every epoch for visualization.

The algorithm is intentionally written in plain Python (no numpy) so the API can
return, for each epoch, the exact distances and cluster assignment of every point
- the same information you would write out by hand when solving a k-means exercise.

Points are represented internally as tuples of floats so the code handles 1-D
(e.g. ``2, 5, 10``) and N-D data uniformly. Distance is Euclidean, which reduces
to absolute difference in one dimension.
"""

import math
import random
from typing import List, Optional, Sequence, Union

Number = Union[int, float]
Point = List[float]


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal-length vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _centroid(points: List[Point]) -> Point:
    """Component-wise mean of a non-empty list of points."""
    dims = len(points[0])
    return [sum(p[d] for p in points) / len(points) for d in range(dims)]


def _as_points(data: Sequence[Union[Number, Sequence[Number]]]) -> List[Point]:
    """Normalize raw 1-D scalars or N-D sequences into a list of float vectors."""
    points: List[Point] = []
    for raw in data:
        if isinstance(raw, (list, tuple)):
            points.append([float(x) for x in raw])
        else:
            points.append([float(raw)])
    return points


def _round(value, ndigits: int = 4):
    """Round scalars or vectors for clean JSON output."""
    if isinstance(value, list):
        return [round(v, ndigits) for v in value]
    return round(value, ndigits)


def generate_data(
    n_points: int = 9,
    dims: int = 1,
    low: float = 0.0,
    high: float = 50.0,
    seed: Optional[int] = None,
    integer: bool = True,
) -> List[List[float]]:
    """Generate random data points for clustering.

    Args:
        n_points: how many points to create.
        dims: dimensionality of each point (1 for the classic number-line case).
        low, high: inclusive range each coordinate is drawn from.
        seed: optional RNG seed for reproducible output.
        integer: if True, draw whole numbers (nicer for hand-checkable demos).

    Returns:
        A list of points. 1-D points are returned as bare numbers so the output
        reads like ``[2, 5, 10]`` rather than ``[[2], [5], [10]]``.
    """
    rng = random.Random(seed)

    def draw() -> float:
        return rng.randint(int(low), int(high)) if integer else rng.uniform(low, high)

    if dims == 1:
        return [draw() for _ in range(n_points)]
    return [[draw() for _ in range(dims)] for _ in range(n_points)]


def _initial_centroids(
    points: List[Point],
    k: int,
    init_centroids: Optional[Sequence[Union[Number, Sequence[Number]]]],
    seed: Optional[int],
) -> List[Point]:
    """Resolve the starting centroids from explicit seeds or random sampling."""
    if init_centroids is not None:
        seeds = _as_points(init_centroids)
        if len(seeds) != k:
            raise ValueError(
                f"init_centroids has {len(seeds)} seeds but k={k} clusters requested."
            )
        dims = len(points[0])
        if any(len(s) != dims for s in seeds):
            raise ValueError("Every seed must have the same dimension as the data.")
        return seeds

    # No explicit seeds: sample k distinct points as the starting centroids.
    rng = random.Random(seed)
    return [list(p) for p in rng.sample(points, k)]


def kmeans(
    data: Sequence[Union[Number, Sequence[Number]]],
    k: int = 2,
    init_centroids: Optional[Sequence[Union[Number, Sequence[Number]]]] = None,
    max_epochs: int = 100,
    seed: Optional[int] = None,
) -> dict:
    """Run k-means and return a step-by-step trace for visualization.

    Convergence is reached when cluster assignments are identical in two
    successive epochs, or when ``max_epochs`` is hit - whichever comes first.

    Args:
        data: the points to cluster, as 1-D scalars or N-D sequences.
        k: number of clusters.
        init_centroids: optional explicit starting centroids (one per cluster).
            If omitted, k points are sampled at random.
        max_epochs: hard cap on the number of epochs.
        seed: RNG seed used only when sampling random centroids.

    Returns:
        A dict containing the (normalized) ``data``, ``initial_centroids``, the
        full ``epochs`` trace, ``converged``, ``epochs_needed`` and the
        ``final_clusters`` with their centroids.
    """
    points = _as_points(data)
    if len(points) < k:
        raise ValueError(f"Need at least k={k} points, got {len(points)}.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    dims = len(points[0])
    if any(len(p) != dims for p in points):
        raise ValueError("All data points must have the same dimension.")

    centroids = _initial_centroids(points, k, init_centroids, seed)
    start_centroids = [list(c) for c in centroids]  # keep for the response echo

    epochs: List[dict] = []
    prev_assignments: Optional[List[int]] = None
    converged = False

    for epoch_num in range(1, max_epochs + 1):
        assignments: List[int] = []
        point_rows = []

        # --- Assignment step: each point joins its nearest centroid. ---
        for idx, point in enumerate(points):
            dists = [_euclidean(point, c) for c in centroids]
            # Ties resolve to the lowest-index centroid (standard convention).
            cluster = min(range(k), key=lambda c: dists[c])
            assignments.append(cluster)
            point_rows.append(
                {
                    "index": idx,
                    "point": _round(point[0] if dims == 1 else point),
                    "distances": [_round(d) for d in dists],
                    "cluster": cluster,
                }
            )

        # --- Update step: recompute each centroid as its members' mean. ---
        clusters: List[dict] = []
        new_centroids: List[Point] = []
        for c in range(k):
            members = [points[i] for i in range(len(points)) if assignments[i] == c]
            if members:
                new_c = _centroid(members)
            else:
                # Empty cluster: keep its centroid where it was (no members to move it).
                new_c = list(centroids[c])
            new_centroids.append(new_c)
            clusters.append(
                {
                    "cluster": c,
                    "indices": [i for i in range(len(points)) if assignments[i] == c],
                    "points": [
                        _round(points[i][0] if dims == 1 else points[i])
                        for i in range(len(points))
                        if assignments[i] == c
                    ],
                    "centroid": _round(new_c[0] if dims == 1 else new_c),
                }
            )

        changed = prev_assignments is None or assignments != prev_assignments
        epochs.append(
            {
                "epoch": epoch_num,
                "centroids": [_round(c[0] if dims == 1 else c) for c in centroids],
                "points": point_rows,
                "clusters": clusters,
                "new_centroids": [
                    _round(c[0] if dims == 1 else c) for c in new_centroids
                ],
                "changed": changed,
            }
        )

        # Convergence: assignments unchanged from the previous epoch.
        if prev_assignments is not None and assignments == prev_assignments:
            converged = True
            break

        prev_assignments = assignments
        centroids = new_centroids

    final = epochs[-1]
    return {
        "data": [_round(p[0] if dims == 1 else p) for p in points],
        "dims": dims,
        "k": k,
        "initial_centroids": [
            _round(c[0] if dims == 1 else c) for c in start_centroids
        ],
        "epochs": epochs,
        "converged": converged,
        "epochs_needed": len(epochs),
        "final_clusters": final["clusters"],
    }
