# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FastAPI backend for NeMo Agent Toolkit Optimizer UI.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from optimizer_ui.backend.api import config_routes
from optimizer_ui.backend.api import optimization_routes
from optimizer_ui.backend.api import results_routes
from optimizer_ui.backend.services.optimization_service import OptimizationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global optimization service
optimization_service = OptimizationService()


@asynccontextmanager
async def lifespan(app: FastAPI):
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(config_routes.router, prefix="/api/config", tags=["config"])
app.include_router(optimization_routes.router, prefix="/api/optimization", tags=["optimization"])
app.include_router(results_routes.router, prefix="/api/results", tags=["results"])

# Serve static files (frontend)
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")


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
