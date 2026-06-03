# MNIST Classifier Training Guide

## Project Overview

This folder contains training and prediction code for the MNIST digit classifier. The model is a simple Multi-Layer Perceptron trained on the MNIST dataset with an option to fine-tune with custom handwritten samples.

## Project Structure

```
mnist_classifier/
├── src/
│   ├── preprocess.py            # Preprocessing (normalization, centering)
│   ├── predict.py               # Inference code
│   ├── retrain_open4s.py        # Fine-tuning script with custom data
│   └── number_four.py           # Data preparation script for custom samples
├── models/
│   ├── mnist.pkl                # Default model (loaded by API)
│   ├── mnist_v1.pkl             # Previous version (backup)
│   └── my_open4s/               # Custom handwritten sample images
├── notebooks/
│   └── MNist.ipynb              # Jupyter notebook with EDA & experiments
├── data/
│   ├── README.md                # Instructions for downloading MNIST data
│   ├── train-images-idx3-ubyte/ # MNIST training images (binary)
│   ├── train-labels-idx1-ubyte/ # MNIST training labels (binary)
│   ├── t10k-images-idx3-ubyte/  # MNIST test images (binary)
│   └── t10k-labels-idx1-ubyte/  # MNIST test labels (binary)
└── TRAINING.md                  # This file
```

## Model Architecture

```
Input (784 features)
  ↓
Hidden Layer 1 (300 units, ReLU)
  ↓
Hidden Layer 2 (100 units, ReLU)
  ↓
Output Layer (10 units, softmax)
```

## Training from Scratch

### 1. Download MNIST Data

```bash
cd data
# Download from http://yann.lecun.com/exdb/mnist/
# Unzip files into respective folders
```

### 2. Train Model

Run the Jupyter notebook `notebooks/MNist.ipynb` to:
- Load MNIST data
- Preprocess images (normalize, center)
- Train the MLP classifier
- Evaluate accuracy
- Save the model

Or create a Python script:
```python
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib

# Load data (from MNist.ipynb or Keras)
X_train = ...  # shape: (60000, 784)
y_train = ...  # shape: (60000,)

# Train
model = MLPClassifier(
    hidden_layer_sizes=(300, 100),
    activation="relu",
    solver="adam",
    max_iter=20,
    verbose=True
)
model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/mnist.pkl")
```

## Fine-tuning with Custom Data

You have two custom digits drawn in Pixil Art that you want to use to fine-tune the "4" detection:

### 1. Prepare Custom Images

Place your handwritten digit images in `models/my_open4s/`:
```
models/
└── my_open4s/
    ├── pixil-frame-0 (1).png
    ├── pixil-frame-0 (2).png
    └── ... (60 custom 4s)
```

Images should be:
- Grayscale PNG or similar
- Any size (will be resized to 28×28)
- One digit per image

### 2. Run Fine-tuning

```bash
cd projects/mnist_classifier
python src/retrain_open4s.py
```

This script:
1. Loads original MNIST training data from `data/`
2. Loads your custom images from `models/my_open4s/`
3. Merges datasets
4. Fine-tunes the model (20 epochs)
5. Saves as `models/mnist_retrained.pkl`

### 3. Test the Model

Update `api_server/routes/mnist.py`:
```python
mnist_model = load_model("mnist_classifier", "mnist_retrained")
```

Then test the API:
```bash
curl -X POST http://localhost:8000/predict/mnist \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0]*784}'
```

## Monitoring & Validation

### Using the Notebook

The `MNist.ipynb` notebook includes:
- Data loading & visualization
- Preprocessing pipeline testing
- Model evaluation on test set
- Confusion matrix
- Sample predictions

### Command Line

```python
from sklearn.metrics import accuracy_score
import numpy as np

# Load test data
test_images = ...  # shape: (10000, 784)
test_labels = ...  # shape: (10000,)

# Evaluate
preds = model.predict(test_images)
acc = accuracy_score(test_labels, preds)
print(f"Accuracy: {acc:.4f}")
```

## Troubleshooting

### "Model file not found"
- Ensure `data/` folder has the MNIST binary files (see `data/README.md`)
- Or update `retrain_open4s.py` to use Keras for MNIST

### Custom images not loading in `retrain_open4s.py`
- Check image names match the regex: `pixil-frame-0 (\d+).png`
- Ensure images are in `models/my_open4s/` folder
- Try running `src/number_four.py` first to prep the data

### Poor accuracy on custom digits
- Collect more custom samples (60+ images per digit)
- Ensure preprocessing in `src/preprocess.py` matches training
- Use more training epochs in `retrain_open4s.py`

## Next Steps

1. **Gather more custom data** — collect 100+ examples per digit
2. **Experiment with architectures** — try deeper networks, dropout, batch norm
3. **Add other digits/models** — create `projects/cifar10_classifier/` for image classification
4. **Evaluate on production data** — test on real user drawings from your portfolio
