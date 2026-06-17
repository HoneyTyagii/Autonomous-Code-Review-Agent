"""FastAPI application entry point with lifespan management.

This module creates the FastAPI app, configures logging, and manages
the application lifecycle (startup/shutdown) for resources like
database connections and background workers.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from code_review_agent.config import get_settings
from code_review_agent.logging import setup_logging, get_logger

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle.

    Handles startup initialization and graceful shutdown of:
    - Logging configuration
    - Database connection pool
    - Redis connection
    - Celery worker connections
    - Vector store client

    Args:
        app: The FastAPI application instance.
    """
    settings = get_settings()

    # --- Startup ---
    setup_logging(
        log_level=settings.log_level,
        json_output=settings.is_production,
    )
    logger.info(
        "starting application",
        environment=settings.app_env.value,
        host=settings.app_host,
        port=settings.app_port,
    )

    # Store settings in app state for access in routes
    app.state.settings = settings

    # TODO: Initialize database pool
    # TODO: Initialize Redis connection
    # TODO: Initialize ChromaDB client
    # TODO: Initialize GitHub App client

    logger.info("application started successfully")

    yield

    # --- Shutdown ---
    logger.info("shutting down application")

    # TODO: Close database pool
    # TODO: Close Redis connection
    # TODO: Cleanup resources

    logger.info("application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Autonomous Code Review Agent",
        description="AI-powered code review agent for GitHub pull requests",
        version="0.1.0",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes on the application.

    Args:
        app: The FastAPI application instance.
    """
    from code_review_agent.api.health import router as health_router

    app.include_router(health_router)

    # TODO: Register webhook router
    # TODO: Register review API router


# Application instance for uvicorn
app = create_app()


def main() -> None:
    """Run the application using uvicorn.

    This is the CLI entry point defined in pyproject.toml.
    """
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "code_review_agent.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
