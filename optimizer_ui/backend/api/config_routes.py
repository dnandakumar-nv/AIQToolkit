# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration management API routes.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nat.data_models.config import Config
from nat.profiler.parameter_optimization.optimizable_utils import walk_optimizables
from nat.runtime.loader import load_config

logger = logging.getLogger(__name__)
router = APIRouter()

# Temporary directory for uploaded configs
UPLOAD_DIR = Path("optimizer_ui_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ConfigResponse(BaseModel):
    """Response model for config data."""
    config_path: str
    config_filename: str
    optimizable_params: dict[str, Any]
    num_numeric_params: int
    num_prompt_params: int


@router.post("/load")
async def load_config_file(file: UploadFile):
    """
    Load a configuration file and extract optimizable parameters.
    Saves the file to disk and returns the path (no serialization/deserialization).
    """
    logger.info(f"Received file upload request: {file.filename}")
    try:
        # Save uploaded file to disk
        content = await file.read()
        logger.info(f"Read {len(content)} bytes from file")

        # Save to upload directory with original filename
        safe_filename = Path(file.filename).name  # Get just the filename, no path
        upload_path = UPLOAD_DIR / safe_filename

        with upload_path.open("wb") as f:
            f.write(content)

        logger.info(f"Saved uploaded file to {upload_path}")

        # Validate the config by loading it
        config_obj: Config = load_config(config_file=upload_path)
        logger.info("Config loaded and validated successfully")

        # Extract optimizable parameters for display only
        optimizable_params = walk_optimizables(config_obj)

        # Convert SearchSpace objects to dicts for JSON serialization
        # Ensure all values are JSON-serializable
        def safe_convert(val):
            """Convert value to JSON-serializable type."""
            if val is None:
                return None
            if isinstance(val, (str, bool)):
                return val
            if isinstance(val, (int, float)):
                return float(val) if isinstance(val, float) else int(val)
            if isinstance(val, Path):
                return str(val)
            return str(val)  # Fallback: convert to string

        optimizable_params_dict = {}
        for key, value in optimizable_params.items():
            try:
                param_dict = {
                    "values": [safe_convert(v) for v in value.values] if value.values is not None else None,
                    "low": safe_convert(value.low),
                    "high": safe_convert(value.high),
                    "log": bool(value.log) if value.log is not None else False,
                    "step": safe_convert(value.step),
                    "is_prompt": bool(value.is_prompt),
                    "prompt": str(value.prompt) if value.prompt is not None else None,
                    "prompt_purpose": str(value.prompt_purpose) if value.prompt_purpose is not None else None,
                }
                optimizable_params_dict[key] = param_dict
            except Exception as e:
                logger.error(f"Error converting parameter {key}: {e}")
                # Skip this parameter if conversion fails
                continue

        # Count param types
        num_numeric = sum(1 for p in optimizable_params.values() if not p.is_prompt)
        num_prompt = sum(1 for p in optimizable_params.values() if p.is_prompt)

        logger.info(f"Loaded config with {len(optimizable_params_dict)} optimizable parameters ({num_numeric} numeric, {num_prompt} prompt)")

        response_data = {
            "config_path": str(upload_path.absolute()),
            "config_filename": safe_filename,
            "optimizable_params": optimizable_params_dict,
            "num_numeric_params": num_numeric,
            "num_prompt_params": num_prompt,
        }

        # Test JSON serialization
        try:
            import json
            json_str = json.dumps(response_data)
            logger.info(f"Response serialized successfully ({len(json_str)} bytes)")
        except Exception as e:
            logger.error(f"Failed to serialize response: {e}")
            raise HTTPException(status_code=500, detail=f"Response serialization error: {str(e)}")

        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading config: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error loading config: {str(e)}")


@router.post("/load-from-path")
async def load_config_from_path(path: str):
    """
    Load a configuration file from a file system path.
    Returns the path (no serialization/deserialization).
    """
    try:
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {path}")

        # Validate the config by loading it
        config_obj: Config = load_config(config_file=config_path)
        logger.info("Config loaded and validated successfully")

        # Extract optimizable parameters for display only
        optimizable_params = walk_optimizables(config_obj)

        # Convert SearchSpace objects to dicts
        # Ensure all values are JSON-serializable
        def safe_convert(val):
            """Convert value to JSON-serializable type."""
            if val is None:
                return None
            if isinstance(val, (str, bool)):
                return val
            if isinstance(val, (int, float)):
                return float(val) if isinstance(val, float) else int(val)
            if isinstance(val, Path):
                return str(val)
            return str(val)  # Fallback: convert to string

        optimizable_params_dict = {}
        for key, value in optimizable_params.items():
            try:
                param_dict = {
                    "values": [safe_convert(v) for v in value.values] if value.values is not None else None,
                    "low": safe_convert(value.low),
                    "high": safe_convert(value.high),
                    "log": bool(value.log) if value.log is not None else False,
                    "step": safe_convert(value.step),
                    "is_prompt": bool(value.is_prompt),
                    "prompt": str(value.prompt) if value.prompt is not None else None,
                    "prompt_purpose": str(value.prompt_purpose) if value.prompt_purpose is not None else None,
                }
                optimizable_params_dict[key] = param_dict
            except Exception as e:
                logger.error(f"Error converting parameter {key}: {e}")
                # Skip this parameter if conversion fails
                continue

        # Count param types
        num_numeric = sum(1 for p in optimizable_params.values() if not p.is_prompt)
        num_prompt = sum(1 for p in optimizable_params.values() if p.is_prompt)

        logger.info(f"Loaded config from {path} with {len(optimizable_params_dict)} optimizable parameters ({num_numeric} numeric, {num_prompt} prompt)")

        response_data = {
            "config_path": str(config_path.absolute()),
            "config_filename": config_path.name,
            "optimizable_params": optimizable_params_dict,
            "num_numeric_params": num_numeric,
            "num_prompt_params": num_prompt,
        }

        # Test JSON serialization
        try:
            import json
            json_str = json.dumps(response_data)
            logger.info(f"Response serialized successfully ({len(json_str)} bytes)")
        except Exception as e:
            logger.error(f"Failed to serialize response: {e}")
            raise HTTPException(status_code=500, detail=f"Response serialization error: {str(e)}")

        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading config from path: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error loading config: {str(e)}")


@router.get("/view/{config_id}")
async def view_config_raw(config_id: str):
    """
    Get the raw YAML content of a config file for viewing (read-only).
    """
    try:
        # For uploaded files
        config_path = UPLOAD_DIR / config_id

        if not config_path.exists():
            # Try as absolute path
            config_path = Path(config_id)
            if not config_path.exists():
                raise HTTPException(status_code=404, detail=f"Config file not found: {config_id}")

        with config_path.open("r") as f:
            content = f.read()

        return {
            "content": content,
            "path": str(config_path),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading config: {str(e)}")
