"""Main FastAPI application for CUBE server."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cube import __version__
from cube.server.config import config
from cube.server.middleware import (
    CUBEException,
    cube_exception_handler,
    general_exception_handler,
    request_logging_middleware,
)
from cube.apis import HealthResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CUBE Standard API",
    description="Common Unified Benchmark Environments - API for benchmark execution",
    version=__version__,
    docs_url="/docs" if config.is_development else None,
    redoc_url="/redoc" if config.is_development else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=config.cors_credentials,
    allow_methods=config.cors_methods,
    allow_headers=config.cors_headers,
)

# Add request logging middleware
app.middleware("http")(request_logging_middleware)

# Register exception handlers
app.add_exception_handler(CUBEException, cube_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


# =============================================================================
# Health Check Endpoint
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check server health and status."""
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=config.environment,
    )


# =============================================================================
# Startup and Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on server startup."""
    logger.info("=" * 80)
    logger.info(f"Starting CUBE Server v{__version__}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Host: {config.host}:{config.port}")
    logger.info(f"Log Level: {config.log_level}")
    logger.info("=" * 80)

    # TODO: Initialize benchmark loader (Dev 2)
    # TODO: Load configured benchmark if specified (Dev 2)

    logger.info("Server startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on server shutdown."""
    logger.info("Shutting down CUBE Server...")

    # TODO: Cleanup all active sessions (Dev 2)
    # TODO: Close benchmark resources (Dev 2)

    logger.info("Server shutdown complete")


# =============================================================================
# Route Registration
# =============================================================================

# TODO: Register benchmark routes (Dev 2)
# from cube.server.routes import benchmark
# app.include_router(benchmark.router, prefix="/cube", tags=["Benchmark"])

# TODO: Register task routes (Dev 3)
# from cube.server.routes import task
# app.include_router(task.router, prefix="/sessions", tags=["Task"])


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CUBE Standard API",
        "version": __version__,
        "docs": f"http://{config.host}:{config.port}/docs" if config.is_development else None,
        "health": f"http://{config.host}:{config.port}/health",
    }
