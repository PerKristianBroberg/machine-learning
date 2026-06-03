import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib, struct, glob, re
from PIL import Image

# ------------------------------------------------------------------
# 1️⃣  Functions to load MNIST from ubyte files
# ------------------------------------------------------------------
def load_images(filename):
    with open(filename, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num, rows * cols)
    return images

def load_labels(filename):
    with open(filename, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# ------------------------------------------------------------------
# 2️⃣  Load original MNIST data
# ------------------------------------------------------------------
train_images = load_images("train-images-idx3-ubyte/train-images-idx3-ubyte")
train_labels = load_labels("train-labels-idx1-ubyte/train-labels-idx1-ubyte")

# Keep same scaling as original model (0–1 floats)
X_train = train_images.astype(np.float32) / 255.0
y_train = train_labels
print("✅ MNIST loaded:", X_train.shape)

# ------------------------------------------------------------------
# 3️⃣  Load custom handwritten sample images
# ------------------------------------------------------------------
files = sorted(glob.glob("custom_samples/pixil-frame-0 (*.png)"))
if not files:
    import os
    files = [f"custom_samples/{f}" for f in os.listdir("custom_samples")
             if re.match(r"pixil-frame-0 \(\d+\)\.png", f)]
print(f"✅ Found {len(files)} custom 4 images")

X_open4 = []
for f in files:
    img = Image.open(f).convert("L")  # grayscale
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(-1)  # flatten to 784
    X_open4.append(arr)
X_open4 = np.array(X_open4)
y_open4 = np.full((len(X_open4),), 4, dtype=np.uint8)

print("Added custom 4s:", X_open4.shape)

# ------------------------------------------------------------------
# 4️⃣  Merge datasets
# ------------------------------------------------------------------
X_combined = np.concatenate([X_train, X_open4])
y_combined = np.concatenate([y_train, y_open4])
print("Total training samples:", X_combined.shape[0])

# ------------------------------------------------------------------
# 5️⃣  Retrain model (same architecture)
# ------------------------------------------------------------------
print("🚀 Fine-tuning MLP with your open-4s...")
model = MLPClassifier(
    hidden_layer_sizes=(300, 100),
    activation="relu",
    solver="adam",
    learning_rate_init=0.0005,
    alpha=1e-4,
    batch_size=128,
    max_iter=20,        # fewer epochs – just fine-tuning
    verbose=True,
    random_state=42
)

model.fit(X_combined, y_combined)
joblib.dump(model, "mnistmlp_open4.pkl")
print("💾 Fine-tuned model saved as mnistmlp_open4.pkl")
