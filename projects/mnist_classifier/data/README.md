# MNIST Dataset

This folder stores training and test data for the MNIST classifier.

## Getting the Data

### Option 1: Download MNIST Binary Files

The original MNIST dataset in binary format (.ubyte) is needed for retraining:

```bash
# Create the folders
mkdir -p train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

# Download from http://yann.lecun.com/exdb/mnist/
# Files to download:
# - train-images-idx3-ubyte.gz → train-images-idx3-ubyte/train-images-idx3-ubyte
# - train-labels-idx1-ubyte.gz → train-labels-idx1-ubyte/train-labels-idx1-ubyte
# - t10k-images-idx3-ubyte.gz → t10k-images-idx3-ubyte/t10k-images-idx3-ubyte
# - t10k-labels-idx1-ubyte.gz → t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte
```

### Option 2: Use Keras/TensorFlow

Alternative: modify `retrain_open4s.py` to load MNIST from Keras:
```python
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
X_train = train_images.reshape(train_images.shape[0], -1).astype(np.float32) / 255.0
y_train = train_labels
```

## Retraining with Custom Data

Place custom handwritten images in `my_open4s/` folder, then:

```bash
python src/retrain_open4s.py
```

This will:
1. Load original MNIST training data
2. Add your custom samples
3. Fine-tune the model
4. Save as `../models/mnist_retrained.pkl`
