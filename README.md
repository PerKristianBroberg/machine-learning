# ML Backend Monorepo

A scalable, modular machine learning backend serving multiple models for the portfolio.

## Structure

```
├── api_server/          # FastAPI entry point, routes, CORS config
├── projects/            # Independent ML projects
│   └── mnist_classifier/
│       ├── notebooks/   # Jupyter notebooks for EDA & experimentation
│       ├── src/         # Modular Python code (preprocess, predict)
│       └── models/      # Serialized model binaries (.pkl)
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

## Development

Use Jupyter notebooks in `projects/{project}/notebooks/` for experimentation and EDA.
