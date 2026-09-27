"""SpatioAI Phase 9 FastAPI application."""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import APISettings
from api.routes import downscaling, ensemble, events, health, risk, tracking
from api.services.model_registry import ModelRegistry
from api.services.repository import DemoRepository
from src.utils.logger import get_logger

logger = get_logger("SpatioAIAPI")


def create_app(settings: APISettings | None = None) -> FastAPI:
    api_settings = settings or APISettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry = ModelRegistry(api_settings)
        registry.load_all()
        app.state.settings = api_settings
        app.state.registry = registry
        app.state.repository = DemoRepository()
        logger.info("SpatioAI API started; model statuses=%s", registry.statuses)
        yield
        logger.info("SpatioAI API stopped")

    application = FastAPI(
        title="SpatioAI API",
        version="0.9.0",
        description="Inference integration layer for the SpatioAI synthetic/experimental ML pipeline.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request failed id=%s path=%s", request_id, request.url.path)
            response = JSONResponse(status_code=500, content={"detail": "internal server error"})
        response.headers["X-Request-ID"] = request_id
        logger.info("request id=%s method=%s path=%s status=%s duration_ms=%.2f", request_id, request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
        return response

    @application.get("/", tags=["system"], summary="API root")
    def root():
        return {"service": "SpatioAI", "version": "0.9.0", "api_version": "v1", "docs": "/docs"}

    application.include_router(health.router)
    application.include_router(events.router)
    application.include_router(tracking.router)
    application.include_router(downscaling.router)
    application.include_router(ensemble.router)
    application.include_router(risk.router)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
