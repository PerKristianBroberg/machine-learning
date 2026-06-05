from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_server.routes import mnist, tic_tac_toe, kmeans

app = FastAPI(title="ML Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pkbroberg.no",
        "https://www.pkbroberg.no",
        # Note: localhost:3000 removed - only for local dev, not production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mnist.router)
app.include_router(tic_tac_toe.router)
app.include_router(kmeans.router)


@app.get("/")
def health():
    return {"status": "running"}
