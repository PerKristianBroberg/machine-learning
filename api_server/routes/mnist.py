import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List
from shared_utils.model_loader import load_model
from projects.mnist_classifier.src.inference import predict_mnist

logger = logging.getLogger(__name__)

router = APIRouter()

# MNIST input is a flattened 28x28 grayscale image, values 0-255.
PIXEL_COUNT = 28 * 28
MAX_PIXEL_VALUE = 255

mnist_model = load_model("mnist_classifier", "mnist_production")


class CanvasInput(BaseModel):
    # Pin the length so oversized payloads are rejected at validation, before
    # the array is ever handed to the model.
    pixels: List[int] = Field(..., min_length=PIXEL_COUNT, max_length=PIXEL_COUNT)

    @field_validator("pixels")
    @classmethod
    def pixels_in_range(cls, value: List[int]) -> List[int]:
        if any(p < 0 or p > MAX_PIXEL_VALUE for p in value):
            raise ValueError(f"pixel values must be between 0 and {MAX_PIXEL_VALUE}")
        return value


@router.post("/predict/mnist")
def predict_mnist_endpoint(input_data: CanvasInput):
    try:
        return predict_mnist(mnist_model, input_data.pixels)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("MNIST prediction failed")
        raise HTTPException(status_code=500, detail="Model prediction failed")
