from fastapi import FastAPI

from api.health import router as health_router
from api.routes import router as mom_router
from common.middleware import process_time_middleware
from config.settings import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION
)

app.middleware("http")(process_time_middleware)

app.include_router(health_router)
app.include_router(mom_router)


@app.get("/")
def root():
    return {
        "message": "RND MOM Generator API is running"
    }