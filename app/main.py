from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import install_error_handlers
from app.api.v1.router import api_router
from app.core.config import settings
from app.sessions.glove import start_glove_feed, stop_glove_feed


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    start_glove_feed()  # solo con GLOVE_LIVE=true
    yield
    stop_glove_feed()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_error_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
