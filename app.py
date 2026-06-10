from fastapi import FastAPI

app = FastAPI(
    title="RND MOM Generator",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "RND MOM Generator is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }