# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Results and visualization API routes.
"""

import base64
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import FileResponse

from optimizer_ui.backend.services import get_optimization_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Get the shared optimization service instance
optimization_service = get_optimization_service()


@router.get("/{run_id}/summary")
async def get_results_summary(run_id: str) -> dict[str, Any]:
    """
    Get a summary of optimization results.
    """
    try:
        # Get the actual output path from the optimization service
        run_status = optimization_service.get_status(run_id)
        if not run_status or "result_path" not in run_status:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        output_dir = Path(run_status["result_path"])

        if not output_dir.exists():
            raise HTTPException(status_code=404, detail=f"Results for run {run_id} not found")

        results = {
            "run_id": run_id,
            "output_path": str(output_dir),
            "files": [],
        }

        # Check for trials dataframe
        trials_csv = output_dir / "trials_dataframe_params.csv"
        if trials_csv.exists():
            df = pd.read_csv(trials_csv)
            results["total_trials"] = len(df)
            results["trials_data"] = df.to_dict(orient="records")

        # Check for optimized config
        optimized_config = output_dir / "optimized_config.yml"
        if optimized_config.exists():
            with optimized_config.open() as f:
                import yaml
                results["optimized_config"] = yaml.safe_load(f)

        # List all files in the output directory
        results["files"] = [str(f.relative_to(output_dir)) for f in output_dir.rglob("*") if f.is_file()]

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting results summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")


@router.get("/{run_id}/trials")
async def get_trials_data(run_id: str) -> dict[str, Any]:
    """
    Get detailed trials data for visualization.
    """
    try:
        # Get the actual output path from the optimization service
        run_status = optimization_service.get_status(run_id)
        if not run_status or "result_path" not in run_status:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        output_dir = Path(run_status["result_path"])
        trials_csv = output_dir / "trials_dataframe_params.csv"

        if not trials_csv.exists():
            raise HTTPException(status_code=404, detail=f"Trials data for run {run_id} not found")

        df = pd.read_csv(trials_csv)

        # Extract value columns for metrics
        value_cols = [col for col in df.columns if col.startswith("values_")]

        # Prepare data for frontend
        trials_data = {
            "trials": df.to_dict(orient="records"),
            "metric_columns": value_cols,
            "param_columns": [col for col in df.columns if col.startswith("params_")],
            "total_trials": len(df),
        }

        # Calculate statistics
        if value_cols:
            trials_data["statistics"] = {
                col: {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
                for col in value_cols
            }

        return trials_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trials data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting trials data: {str(e)}")


@router.get("/{run_id}/visualizations")
async def get_visualizations(run_id: str) -> dict[str, Any]:
    """
    Get all available visualizations for a run.
    """
    try:
        # Get the actual output path from the optimization service
        run_status = optimization_service.get_status(run_id)
        if not run_status or "result_path" not in run_status:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        output_dir = Path(run_status["result_path"])
        plots_dir = output_dir / "plots"

        if not output_dir.exists():
            raise HTTPException(status_code=404, detail=f"Results for run {run_id} not found")

        visualizations = {}

        # Check for plot files
        if plots_dir.exists():
            for plot_file in plots_dir.glob("*.png"):
                with plot_file.open("rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
                    visualizations[plot_file.stem] = {
                        "name": plot_file.stem,
                        "type": "image",
                        "data": f"data:image/png;base64,{image_data}",
                    }

        return {
            "run_id": run_id,
            "visualizations": visualizations,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visualizations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting visualizations: {str(e)}")


@router.get("/{run_id}/download/{file_type}")
async def download_file(run_id: str, file_type: str):
    """
    Download a specific result file.

    file_type can be: config, trials, pareto_2d, pareto_parallel, pareto_pairwise
    """
    try:
        # Get the actual output path from the optimization service
        run_status = optimization_service.get_status(run_id)
        if not run_status or "result_path" not in run_status:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        output_dir = Path(run_status["result_path"])

        file_map = {
            "config": output_dir / "optimized_config.yml",
            "trials": output_dir / "trials_dataframe_params.csv",
            "pareto_2d": output_dir / "plots" / "pareto_front_2d.png",
            "pareto_parallel": output_dir / "plots" / "pareto_parallel_coordinates.png",
            "pareto_pairwise": output_dir / "plots" / "pareto_pairwise_matrix.png",
        }

        if file_type not in file_map:
            raise HTTPException(status_code=400, detail=f"Unknown file type: {file_type}")

        file_path = file_map[file_type]

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path.name}")

        return FileResponse(
            path=file_path,
            filename=f"{run_id}_{file_path.name}",
            media_type="application/octet-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@router.get("/{run_id}/insights")
async def get_insights(run_id: str) -> dict[str, Any]:
    """
    Get AI-powered insights from optimization results.
    """
    try:
        # Get the actual output path from the optimization service
        run_status = optimization_service.get_status(run_id)
        if not run_status or "result_path" not in run_status:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        output_dir = Path(run_status["result_path"])
        trials_csv = output_dir / "trials_dataframe_params.csv"

        if not trials_csv.exists():
            raise HTTPException(status_code=404, detail=f"Results for run {run_id} not found")

        df = pd.read_csv(trials_csv)

        # Generate insights
        insights = []

        # Find value columns (metrics)
        value_cols = [col for col in df.columns if col.startswith("values_")]

        # Best trial insight
        if value_cols:
            for i, col in enumerate(value_cols):
                best_idx = df[col].idxmax() if col.startswith("values_") else df[col].idxmin()
                best_trial = df.loc[best_idx]
                insights.append({
                    "type": "best_trial",
                    "title": f"Best Trial for {col}",
                    "description": f"Trial {best_trial['number']} achieved the best {col} value of {best_trial[col]:.4f}",
                    "severity": "success",
                })

        # Parameter correlation insights
        param_cols = [col for col in df.columns if col.startswith("params_")]
        if param_cols and value_cols:
            for value_col in value_cols:
                for param_col in param_cols:
                    # Check if numeric
                    if pd.api.types.is_numeric_dtype(df[param_col]):
                        corr = df[param_col].corr(df[value_col])
                        if abs(corr) > 0.5:
                            insights.append({
                                "type": "correlation",
                                "title": f"Parameter Correlation: {param_col}",
                                "description": f"{param_col} has a {'strong positive' if corr > 0 else 'strong negative'} correlation ({corr:.3f}) with {value_col}",
                                "severity": "info",
                            })

        # Convergence insight
        if len(df) >= 10:
            # Check if metrics are improving over time
            for value_col in value_cols:
                early_mean = df[value_col].head(len(df) // 3).mean()
                late_mean = df[value_col].tail(len(df) // 3).mean()
                improvement = ((late_mean - early_mean) / early_mean) * 100

                if abs(improvement) > 5:
                    insights.append({
                        "type": "convergence",
                        "title": f"Optimization Progress for {value_col}",
                        "description": f"{'Improved' if improvement > 0 else 'Declined'} by {abs(improvement):.1f}% from early to late trials",
                        "severity": "info",
                    })

        # Add general statistics
        insights.append({
            "type": "statistics",
            "title": "Optimization Summary",
            "description": f"Completed {len(df)} trials across {len(param_cols)} parameters optimizing {len(value_cols)} metrics",
            "severity": "info",
        })

        return {
            "run_id": run_id,
            "insights": insights,
            "generated_at": pd.Timestamp.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")
