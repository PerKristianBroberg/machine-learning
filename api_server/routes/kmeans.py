import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union

from projects.kmeans.src.engine import generate_data, kmeans

logger = logging.getLogger(__name__)

router = APIRouter()

# A point is either a bare number (1-D) or a list of numbers (N-D).
Scalar = Union[int, float]
PointInput = Union[Scalar, List[Scalar]]

# Caps that mirror the /generate/kmeans limits. Without them, `data` and
# `init_centroids` are unbounded: a large point set crossed with up to
# max_epochs iterations and a full per-epoch trace is a memory/CPU/response
# amplification vector.
MAX_POINTS = 1000
MAX_DIMS = 10


def _check_point_dims(points: Optional[List[PointInput]]) -> Optional[List[PointInput]]:
    """Reject points whose dimensionality exceeds MAX_DIMS."""
    if points is None:
        return points
    for p in points:
        if isinstance(p, list) and len(p) > MAX_DIMS:
            raise ValueError(f"each point may have at most {MAX_DIMS} dimensions")
    return points


class ClusterInput(BaseModel):
    data: Optional[List[PointInput]] = Field(
        None,
        max_length=MAX_POINTS,
        description="Points to cluster: bare numbers for 1-D, or lists for N-D. "
        "If omitted, random data is generated.",
    )
    k: int = Field(2, ge=1, le=10, description="Number of clusters.")
    init_centroids: Optional[List[PointInput]] = Field(
        None,
        max_length=MAX_DIMS,
        description="Optional explicit starting centroids (one per cluster).",
    )
    max_epochs: int = Field(100, ge=1, le=1000, description="Hard cap on epochs.")
    seed: Optional[int] = Field(
        None, description="RNG seed for random centroids / data generation."
    )

    @field_validator("data", "init_centroids")
    @classmethod
    def limit_dims(cls, value: Optional[List[PointInput]]) -> Optional[List[PointInput]]:
        return _check_point_dims(value)


class GenerateInput(BaseModel):
    n_points: int = Field(9, ge=1, le=1000, description="How many points to generate.")
    dims: int = Field(1, ge=1, le=10, description="Dimensionality of each point.")
    low: float = Field(0.0, description="Inclusive lower bound for coordinates.")
    high: float = Field(50.0, description="Inclusive upper bound for coordinates.")
    integer: bool = Field(True, description="Draw whole numbers if True.")
    seed: Optional[int] = Field(None, description="RNG seed for reproducibility.")


@router.post("/predict/kmeans")
def kmeans_cluster(input_data: ClusterInput):
    """Cluster the given (or randomly generated) data and return a per-epoch trace."""
    try:
        data = input_data.data
        if data is None:
            data = generate_data(seed=input_data.seed)
        result = kmeans(
            data,
            k=input_data.k,
            init_centroids=input_data.init_centroids,
            max_epochs=input_data.max_epochs,
            seed=input_data.seed,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Clustering failed")
        raise HTTPException(status_code=500, detail="Clustering failed")


@router.post("/generate/kmeans")
def kmeans_generate(input_data: GenerateInput):
    """Generate random data points the client can then send back for clustering."""
    try:
        return {
            "data": generate_data(
                n_points=input_data.n_points,
                dims=input_data.dims,
                low=input_data.low,
                high=input_data.high,
                seed=input_data.seed,
                integer=input_data.integer,
            )
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Data generation failed")
        raise HTTPException(status_code=500, detail="Data generation failed")
