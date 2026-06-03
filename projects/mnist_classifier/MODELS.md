# MNIST Model Versions

## Production Model: `models/mnist_production.pkl`

**Status:** ✅ In use by the API

**What it includes:**
- 60,000 MNIST training samples
- 60 custom handwritten 4s (drawn in Pixil Art)
- Fine-tuned for 20 iterations

**Performance:**
- Test accuracy: ~98%
- Optimized for recognizing both standard MNIST digits AND your custom handwriting style

**How it was created:**
1. Started with standard MNIST trained model
2. Ran `src/fine_tune_model.py` with your 60 custom 4s
3. Model learned to recognize your specific style of writing "4"

**File size:** 4.1M (smaller than original because fine-tuning converges faster)

---

## Baseline Model: `models/mnist_baseline.pkl`

**Status:** ⚠️ Older version, not currently used

**What it includes:**
- 60,000 MNIST training samples only
- No custom handwritten samples
- Original training from notebook

**Performance:**
- Test accuracy: ~98.33%
- Generic MNIST accuracy (slightly higher because no overfitting to your 4s)

**File size:** 8.2M (larger, trained for full 40 iterations)

---

## How to Update the Model

If you want to retrain with **more custom samples**:

```bash
# 1. Add your custom digit images to:
# data/custom_samples/pixil-frame-0 (*.png)

# 2. Run retraining:
python src/fine_tune_model.py

# 3. This creates: models/mnistmlp_open4.pkl

# 4. Decide:
#    Option A: Overwrite current model
cp models/mnistmlp_open4.pkl models/mnist_production.pkl

#    Option B: Keep both versions
#    (update api_server/routes/mnist.py to load which one)
```

---

## Model Training Pipeline

```
MNIST Original Dataset (60,000)
        ↓
   Train MLP (40 epochs)
        ↓
  mnistmlp.pkl ← (stored as mnist_v1.pkl)
        ↓
   Load trained model
        ↓
   + Your 60 custom 4s
        ↓
   Fine-tune MLP (20 epochs)
        ↓
  mnistmlp_open4.pkl ← (stored as mnist.pkl) ✅ PRODUCTION
```

---

## Switching Models

To use the baseline model without custom 4s:

**Option 1: Swap files**
```bash
mv models/mnist_production.pkl models/mnist_production.pkl.backup
cp models/mnist_baseline.pkl models/mnist_production.pkl
# Restart API
```

**Option 2: Update API route**
```python
# In api_server/routes/mnist.py
mnist_model = load_model("mnist_classifier", "mnist_baseline")
```

---

## Adding Custom Samples for Other Digits

To fine-tune for other digits (not just 4):

1. Create folders: `models/my_0s/`, `models/my_1s/`, etc.
2. Modify `src/retrain_open4s.py` to load from all folders
3. Label them accordingly: `np.full((len(X_0s),), 0)` for 0s, etc.
4. Run retraining and save as `mnist_multiclass.pkl`
