from .preprocessing import preprocess_mnist


def predict_mnist(model, pixels: list) -> dict:
    """Predict MNIST digit from pixel data."""
    arr = preprocess_mnist(pixels)

    prediction = int(model.predict(arr)[0])
    proba = model.predict_proba(arr)[0]
    probabilities = {str(i): float(p) for i, p in enumerate(proba)}

    return {
        "prediction": prediction,
        "probabilities": probabilities
    }
