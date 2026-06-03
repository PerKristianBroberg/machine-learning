from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from shared_utils.model_loader import load_model
from projects.mnist_classifier.src.predict import predict_mnist

router = APIRouter()

mnist_model = load_model("mnist_classifier", "mnist")


class CanvasInput(BaseModel):
    pixels: List[int]


@router.post("/predict/mnist")
def predict_mnist_endpoint(input_data: CanvasInput):
    try:
        return predict_mnist(mnist_model, input_data.pixels)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
