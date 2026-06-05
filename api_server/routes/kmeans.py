from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Union

from projects.kmeans.src.engine import generate_data, kmeans

router = APIRouter()

# A point is either a bare number (1-D) or a list of numbers (N-D).
Scalar = Union[int, float]
PointInput = Union[Scalar, List[Scalar]]


class ClusterInput(BaseModel):
    data: Optional[List[PointInput]] = Field(
        None,
        description="Points to cluster: bare numbers for 1-D, or lists for N-D. "
        "If omitted, random data is generated.",
    )
    k: int = Field(2, ge=1, le=10, description="Number of clusters.")
    init_centroids: Optional[List[PointInput]] = Field(
        None, description="Optional explicit starting centroids (one per cluster)."
    )
    max_epochs: int = Field(100, ge=1, le=1000, description="Hard cap on epochs.")
    seed: Optional[int] = Field(
        None, description="RNG seed for random centroids / data generation."
    )


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {e}")


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation failed: {e}")
