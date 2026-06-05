# ML Backend Monorepo

A scalable, modular machine learning backend serving multiple models for the portfolio.

## Structure

```
├── api_server/          # FastAPI entry point, routes, CORS config
├── projects/            # Independent ML projects
│   └── mnist_classifier/
│       ├── notebooks/   # Jupyter notebooks for EDA & experimentation
│       ├── src/         # Modular Python code (preprocessing, inference, training)
│       ├── models/      # Serialized model binaries (.pkl)
│       └── data/        # Training datasets and custom samples
├── shared_utils/        # Common utilities (model loader, etc)
└── requirements.txt     # Python dependencies
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r api_server/requirements.txt
```

## Running Locally

```bash
uvicorn api_server.app:app --reload
```

Server starts at `http://localhost:8000`

- `GET /` — health check
- `POST /predict/mnist` — predict digit from pixel array
- `POST /predict/tic-tac-toe` — optimal move via alpha-beta minimax
- `POST /predict/kmeans` — cluster 1-D/N-D data, returns a per-epoch trace
- `POST /generate/kmeans` — generate random data points to cluster

### Test k-means endpoint

```bash
# Reproduces the classic exercise: cluster 2,5,10,12,3,20,31,11,24 with seeds 2 & 5
curl -X POST http://localhost:8000/predict/kmeans \
  -H "Content-Type: application/json" \
  -d '{"data": [2,5,10,12,3,20,31,11,24], "k": 2, "init_centroids": [2,5]}'

# Or generate random data first, then cluster it
curl -X POST http://localhost:8000/generate/kmeans \
  -H "Content-Type: application/json" \
  -d '{"n_points": 12, "dims": 1, "low": 0, "high": 50}'
```

The k-means response includes `epochs` — each with every point's distance to each
centroid and its cluster assignment — plus `initial_centroids`, `converged`,
`epochs_needed` and `final_clusters`, giving a frontend everything it needs to
animate the algorithm step by step.

### Test MNIST endpoint

```bash
curl -X POST http://localhost:8000/predict/mnist \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0] * 784}'
```

## Adding a New Model

1. Create a new project folder:
   ```
   projects/my_project/
   ├── notebooks/
   ├── src/
   │   ├── __init__.py
   │   ├── preprocess.py
   │   └── predict.py
   └── models/
       └── my_model.pkl
   ```

2. Create `api_server/routes/my_project.py`:
   ```python
   from fastapi import APIRouter
   from shared_utils.model_loader import load_model
   from projects.my_project.src.predict import predict_my_model

   router = APIRouter()
   model = load_model("my_project", "my_model")

   @router.post("/predict/my_project")
   def my_endpoint(input_data):
       return predict_my_model(model, input_data)
   ```

3. Register the route in `api_server/app.py`:
   ```python
   from api_server.routes import my_project
   app.include_router(my_project.router)
   ```

## Deployment to Railway

- Root directory: `backend-ml/`
- Start command: `uvicorn api_server.app:app --host 0.0.0.0 --port $PORT`
- Environment: Python 3.11+

## Models

**Production:** `projects/mnist_classifier/models/mnist_production.pkl`
- Fine-tuned with 60 custom handwritten 4s
- Ready to use, no setup needed
- See `MODELS.md` for details on versions

## Training & Development

### MNIST Retraining

Fine-tune with more custom samples:

1. **Add images:** Place in `projects/mnist_classifier/data/custom_samples/`
2. **Retrain:**
   ```bash
   cd projects/mnist_classifier
   python src/fine_tune_model.py
   ```
3. **Deploy:** Replace or update model reference in API

See `projects/mnist_classifier/TRAINING.md` and `MODELS.md` for details.

### Notebooks

Use Jupyter notebooks in `projects/{project}/notebooks/` for experimentation and EDA:

```bash
jupyter notebook projects/mnist_classifier/notebooks/mnist_exploration.ipynb
```
