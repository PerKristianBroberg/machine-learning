import numpy as np
from scipy.ndimage import measurements


def preprocess_mnist(pixels: list) -> np.ndarray:
    """Preprocess pixel data for MNIST model."""
    if len(pixels) != 28 * 28:
        raise ValueError("Input must contain 784 pixels (28x28).")

    arr = 255 - np.array(pixels, dtype=np.float32)
    arr /= 255.0
    img = arr.reshape(28, 28)

    cy, cx = measurements.center_of_mass(img)
    if np.isnan(cx) or np.isnan(cy):
        arr = img.reshape(1, 28 * 28)
    else:
        shiftx = int(np.round(14 - cx))
        shifty = int(np.round(14 - cy))
        img = np.roll(img, shiftx, axis=1)
        img = np.roll(img, shifty, axis=0)
        arr = img.reshape(1, 28 * 28)

    return arr
