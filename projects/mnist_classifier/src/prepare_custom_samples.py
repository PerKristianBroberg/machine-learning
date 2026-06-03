from PIL import Image
import numpy as np
import glob
import re

# Path to custom samples folder
files = sorted(glob.glob("custom_samples/pixil-frame-0 (*.png)"))

if not files:
    # Alternative matching if parentheses cause glob issues
    files = sorted(glob.glob("custom_samples/pixil-frame-0*(*).png"))

# Fallback pattern if needed (manual regex)
if not files:
    import os
    files = [f for f in os.listdir("custom_samples") if re.match(r"pixil-frame-0 \(\d+\)\.png", f)]
    files = [f"custom_samples/{f}" for f in files]

print(f"Found {len(files)} images")

X_new = []
for f in files:
    img = Image.open(f).convert("L")  # grayscale
    arr = np.array(img, dtype=np.float32) / 255.0  # normalize to 0–1
    X_new.append(arr)

X_new = np.array(X_new)
y_new = np.full((len(X_new),), 4, dtype=np.uint8)  # all labeled as 4

np.save("X_open4.npy", X_new)
np.save("y_open4.npy", y_new)

print("✅ Saved: X_open4.npy, y_open4.npy")
