from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_server.routes import mnist

app = FastAPI(title="ML Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pkbroberg.no",
        "https://www.pkbroberg.no",
        "http://www.pkbroberg.no",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mnist.router)


@app.get("/")
def health():
    return {"status": "running"}
