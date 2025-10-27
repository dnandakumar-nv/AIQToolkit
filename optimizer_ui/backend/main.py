# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FastAPI backend for NeMo Agent Toolkit Optimizer UI.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from optimizer_ui.backend.api import config_routes
from optimizer_ui.backend.api import optimization_routes
from optimizer_ui.backend.api import results_routes
from optimizer_ui.backend.services import get_optimization_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the shared optimization service instance
optimization_service = get_optimization_service()


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Starting Optimizer UI Backend...")
    yield
    logger.info("Shutting down Optimizer UI Backend...")
    await optimization_service.cleanup()


# Create FastAPI app
app = FastAPI(
    title="NeMo Agent Toolkit Optimizer UI",
    description="Interactive UI for optimizing NeMo Agent Toolkit workflows",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS - allow all origins for development
# WARNING: In production, you should restrict this to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers to the client
)

# Include routers BEFORE static files to ensure API routes take precedence
app.include_router(
    config_routes.router, prefix="/api/config", tags=["config"]
)
app.include_router(
    optimization_routes.router,
    prefix="/api/optimization",
    tags=["optimization"]
)
app.include_router(
    results_routes.router, prefix="/api/results", tags=["results"]
)

# Serve static files (frontend) - mounted LAST
# so API routes take precedence
# Handle both running from optimizer_ui/ and from parent directory
current_dir = Path(__file__).parent
if current_dir.name == "backend":
    # Running from optimizer_ui/backend/
    frontend_path = current_dir.parent / "frontend"
else:
    # Running from parent directory (NAT-Local/)
    frontend_path = Path("optimizer_ui/frontend")

frontend_path = frontend_path.resolve()
if frontend_path.exists():
    logger.info("Serving frontend from %s", frontend_path)
    app.mount(
        "/",
        StaticFiles(directory=str(frontend_path), html=True),
        name="static"
    )
else:
    logger.warning("Frontend path not found: %s", frontend_path)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "optimizer-ui",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
