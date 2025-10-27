# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Optimization execution API routes with WebSocket support.
"""

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from pydantic import BaseModel

from optimizer_ui.backend.services import get_optimization_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Get the shared optimization service instance
optimization_service = get_optimization_service()


class StartOptimizationRequest(BaseModel):
    """Request model for starting optimization."""
    config_path: str  # Changed from config dict to config file path
    dataset_path: str | None = None
    result_json_path: str = "$"
    endpoint: str | None = None
    endpoint_timeout: int = 300


class OptimizationStatusResponse(BaseModel):
    """Response model for optimization status."""
    run_id: str
    status: str
    progress: float
    current_trial: int | None = None
    total_trials: int | None = None
    message: str | None = None
    result_path: str | None = None


@router.post("/start")
async def start_optimization(request: StartOptimizationRequest) -> dict[str, str]:
    """
    Start a new optimization run.
    """
    try:
        # Verify config file exists
        from pathlib import Path
        config_path = Path(request.config_path)
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {request.config_path}")

        run_id = await optimization_service.start_optimization(
            config_path=str(config_path.absolute()),
            dataset_path=request.dataset_path,
            result_json_path=request.result_json_path,
            endpoint=request.endpoint,
            endpoint_timeout=request.endpoint_timeout,
        )

        return {
            "run_id": run_id,
            "message": "Optimization started",
            "status": "running",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting optimization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting optimization: {str(e)}")


@router.get("/status/{run_id}")
async def get_optimization_status(run_id: str) -> OptimizationStatusResponse:
    """
    Get the status of a running or completed optimization.
    """
    status = optimization_service.get_status(run_id)

    if not status:
        raise HTTPException(status_code=404, detail=f"Optimization run {run_id} not found")

    return OptimizationStatusResponse(**status)


@router.post("/stop/{run_id}")
async def stop_optimization(run_id: str) -> dict[str, str]:
    """
    Stop a running optimization.
    """
    success = await optimization_service.stop_optimization(run_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Optimization run {run_id} not found or already stopped")

    return {
        "run_id": run_id,
        "message": "Optimization stopped",
        "status": "stopped",
    }


@router.get("/runs")
async def list_optimization_runs() -> list[dict[str, Any]]:
    """
    List all optimization runs.
    """
    return optimization_service.list_runs()


@router.websocket("/ws/{run_id}")
async def optimization_websocket(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for real-time optimization progress updates.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for run {run_id}")

    try:
        # Subscribe to updates for this run
        queue = optimization_service.subscribe(run_id)

        while True:
            try:
                # Get updates from the queue with timeout
                update = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(update)

                # Check if optimization is complete or failed
                if update.get("status") in ["completed", "failed", "stopped"]:
                    logger.info(f"Optimization {run_id} finished with status: {update.get('status')}")
                    break

            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")
    except Exception as e:
        logger.error(f"WebSocket error for run {run_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except:
            pass
    finally:
        optimization_service.unsubscribe(run_id, queue)
        await websocket.close()
