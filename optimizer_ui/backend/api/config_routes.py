# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration management API routes.
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import UploadFile
from pydantic import BaseModel

from nat.data_models.config import Config
from nat.data_models.optimizable import SearchSpace
from nat.profiler.parameter_optimization.optimizable_utils import walk_optimizables
from nat.runtime.loader import load_config

logger = logging.getLogger(__name__)
router = APIRouter()


class ConfigResponse(BaseModel):
    """Response model for config data."""
    config: dict[str, Any]
    optimizable_params: dict[str, Any]
    config_path: str | None = None


class UpdateConfigRequest(BaseModel):
    """Request model for updating config."""
    config: dict[str, Any]
    config_path: str | None = None


@router.post("/load")
async def load_config_file(file: UploadFile) -> ConfigResponse:
    """
    Load a configuration file and extract optimizable parameters.
    """
    try:
        content = await file.read()
        config_dict = yaml.safe_load(content)

        # Load config using NAT's loader
        config_obj: Config = Config.model_validate(config_dict)

        # Extract optimizable parameters
        optimizable_params = walk_optimizables(config_obj)

        # Convert SearchSpace objects to dicts for JSON serialization
        optimizable_params_dict = {
            key: {
                "values": value.values,
                "low": value.low,
                "high": value.high,
                "log": value.log,
                "step": value.step,
                "is_prompt": value.is_prompt,
                "prompt": value.prompt,
                "prompt_purpose": value.prompt_purpose,
            }
            for key, value in optimizable_params.items()
        }

        return ConfigResponse(
            config=config_obj.model_dump(),
            optimizable_params=optimizable_params_dict,
            config_path=file.filename,
        )
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading config: {str(e)}")


@router.post("/load-from-path")
async def load_config_from_path(path: str) -> ConfigResponse:
    """
    Load a configuration file from a file system path.
    """
    try:
        config_path = Path(path)
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {path}")

        config_obj: Config = load_config(config_file=config_path)

        # Extract optimizable parameters
        optimizable_params = walk_optimizables(config_obj)

        # Convert SearchSpace objects to dicts
        optimizable_params_dict = {
            key: {
                "values": value.values,
                "low": value.low,
                "high": value.high,
                "log": value.log,
                "step": value.step,
                "is_prompt": value.is_prompt,
                "prompt": value.prompt,
                "prompt_purpose": value.prompt_purpose,
            }
            for key, value in optimizable_params.items()
        }

        return ConfigResponse(
            config=config_obj.model_dump(),
            optimizable_params=optimizable_params_dict,
            config_path=str(config_path),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading config from path: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading config: {str(e)}")


@router.post("/save")
async def save_config(request: UpdateConfigRequest) -> dict[str, str]:
    """
    Save the updated configuration to a file.
    """
    try:
        if not request.config_path:
            raise HTTPException(status_code=400, detail="config_path is required")

        config_path = Path(request.config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w") as f:
            yaml.dump(request.config, f, default_flow_style=False, sort_keys=False)

        return {"message": "Config saved successfully", "path": str(config_path)}
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving config: {str(e)}")


@router.post("/validate")
async def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a configuration without saving it.
    """
    try:
        config_obj = Config.model_validate(config)
        optimizable_params = walk_optimizables(config_obj)

        return {
            "valid": True,
            "message": "Configuration is valid",
            "optimizable_param_count": len(optimizable_params),
        }
    except Exception as e:
        logger.error(f"Config validation error: {e}")
        return {
            "valid": False,
            "message": str(e),
        }
