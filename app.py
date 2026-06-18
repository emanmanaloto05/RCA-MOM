from fastapi import FastAPI

from api.routes import router as mom_router

app = FastAPI(
    title="AI Meeting Notes Summarizer",
    version="1.0.0"
)

app.include_router(
    mom_router,
    prefix="/api",
    tags=["Meeting Notes"]
)


@app.get("/")
def root():
    return {
        "message": "AI Meeting Notes Summarizer API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }