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
            # No trials data - this is normal for prompt-only optimization
            return {
                "trials": [],
                "metric_columns": [],
                "param_columns": [],
                "total_trials": 0,
                "message": "No trials data available (prompt-only optimization or no numeric optimization run)"
            }

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
            # No trials data - check if prompt optimization was run
            prompt_files = list(output_dir.glob("optimized_prompts_gen*.json"))
            if prompt_files:
                # Prompt optimization was run
                return {
                    "run_id": run_id,
                    "insights": [{
                        "type": "info",
                        "title": "Prompt Optimization Complete",
                        "description": f"Completed {len(prompt_files)} generations of prompt optimization. No numeric parameter trials data available.",
                        "severity": "info",
                    }],
                    "generated_at": pd.Timestamp.now().isoformat(),
                }
            else:
                return {
                    "run_id": run_id,
                    "insights": [],
                    "message": "No optimization results found",
                    "generated_at": pd.Timestamp.now().isoformat(),
                }

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


@router.get("/{run_id}/prompts")
async def get_prompt_comparisons(run_id: str) -> dict[str, Any]:
    """
    Get prompt comparisons showing before/after optimization.
    Reads from optimized_prompts_gen*.json files.

    File format: {param_name: [prompt_text, purpose]}
    """
    try:
        # Get the actual output path from the optimization service
        run_status = optimization_service.get_status(run_id)
        if not run_status or "result_path" not in run_status:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        output_dir = Path(run_status["result_path"])

        if not output_dir.exists():
            raise HTTPException(status_code=404, detail=f"Results for run {run_id} not found")

        # Find all generation files sorted numerically
        import json
        import re

        prompt_files = list(output_dir.glob("optimized_prompts_gen*.json"))
        if prompt_files:
            # Sort by generation number
            def get_gen_num(path):
                match = re.search(r'gen(\d+)', path.stem)
                return int(match.group(1)) if match else 0
            prompt_files = sorted(prompt_files, key=get_gen_num)

        logger.info(f"Found {len(prompt_files)} prompt generation files for run {run_id}")

        if not prompt_files:
            # No prompt optimization was run
            logger.info(f"No prompt files found in {output_dir}")
            return {
                "run_id": run_id,
                "comparisons": [],
                "message": "No prompt optimization results found",
            }

        # CRITICAL: Get the FINAL optimized prompts from optimized_prompts.json
        # This is the final output after all GA generations, NOT an intermediate gen{n} file
        final_prompts_file = output_dir / "optimized_prompts.json"
        if not final_prompts_file.exists():
            # Fallback: Use the last generation file if final doesn't exist yet
            final_prompts_file = prompt_files[-1]
            logger.warning(f"optimized_prompts.json not found, using last generation file: {final_prompts_file}")
        else:
            logger.info(f"Using FINAL optimized prompts file: {final_prompts_file}")

        # Load FINAL optimized prompts (AFTER optimization)
        with final_prompts_file.open() as f:
            optimized_prompts = json.load(f)
            logger.info(f"Loaded {len(optimized_prompts)} FINAL optimized prompts from {final_prompts_file.name}")

        # Get ORIGINAL prompts from the configuration file (BEFORE any optimization)
        original_prompts = {}
        
        try:
            from nat.runtime.loader import load_config
            from nat.profiler.parameter_optimization.optimizable_utils import walk_optimizables

            config_path = run_status.get("config_path")
            logger.info(f"Loading original prompts from config: {config_path}")
            
            if config_path:
                config_obj = load_config(config_file=Path(config_path))
                full_space = walk_optimizables(config_obj)
                
                # Extract original prompts from the config object
                for param_name, search_space in full_space.items():
                    if search_space.is_prompt:
                        logger.info(f"Processing prompt parameter: {param_name}")
                        
                        # Get the actual field value from the config object
                        try:
                            parts = param_name.split('.')
                            obj = config_obj
                            for part in parts:
                                # Handle both dict and object attribute access
                                if isinstance(obj, dict):
                                    obj = obj[part]
                                else:
                                    obj = getattr(obj, part)
                            
                            original_prompt = obj
                            if original_prompt:
                                logger.info(f"Found original prompt for {param_name}: {original_prompt[:50]}...")
                                original_prompts[param_name] = (original_prompt, search_space.prompt_purpose or "N/A")
                            else:
                                logger.warning(f"Original prompt value is empty for {param_name}")
                        except Exception as e:
                            logger.warning(f"Could not navigate to {param_name}: {e}")
                            # Try using the SearchSpace prompt as fallback
                            if search_space.prompt:
                                logger.info(f"Using SearchSpace.prompt as fallback for {param_name}")
                                original_prompts[param_name] = (search_space.prompt, search_space.prompt_purpose or "N/A")
                
                logger.info(f"Loaded {len(original_prompts)} original prompts from config")
        except Exception as e:
            logger.error(f"Failed to load original prompts from config: {e}", exc_info=True)
        
        if not original_prompts:
            logger.warning("Could not load original prompts from config, falling back to gen1")
            # Fallback: use gen1 as the source of original prompts
            if prompt_files:
                gen1_file = prompt_files[0]
                logger.info(f"Using {gen1_file.name} as fallback source of original prompts")
                with gen1_file.open() as f:
                    gen1_data = json.load(f)
                    original_prompts = gen1_data
                    logger.info(f"Loaded {len(original_prompts)} prompts from gen1")

        # Format comparisons for frontend
        # COMPARISON: ORIGINAL (from config) vs FINAL (from optimized_prompts.json)
        # Structure: {param_name: [prompt_text, purpose]}
        comparisons = []
        for param_name, prompt_data in optimized_prompts.items():
            # Handle both tuple and list formats
            if isinstance(prompt_data, (list, tuple)) and len(prompt_data) >= 2:
                optimized_text, purpose = prompt_data[0], prompt_data[1]
            else:
                logger.warning(f"Unexpected format for {param_name}: {type(prompt_data)}")
                continue

            # Get ORIGINAL prompt from config (BEFORE optimization)
            if param_name in original_prompts:
                orig_data = original_prompts[param_name]
                if isinstance(orig_data, (list, tuple)) and len(orig_data) >= 2:
                    original_text = orig_data[0]
                else:
                    original_text = str(orig_data)  # Handle string case
            else:
                logger.warning(f"Original prompt not found for {param_name}, using optimized as fallback")
                original_text = optimized_text  # Fallback if not found
            
            # Check if the prompt actually changed
            prompt_changed = original_text != optimized_text
            logger.info(f"Comparison for {param_name}: changed={prompt_changed}, original={original_text[:30]}..., optimized={optimized_text[:30]}...")

            # Create comparison: BEFORE (original from config) → AFTER (final from optimized_prompts.json)
            comparisons.append({
                "name": param_name,
                "purpose": purpose,
                "before": original_text,  # Original from config/gen1
                "after": optimized_text,  # Final from optimized_prompts.json
                "changed": prompt_changed,  # Flag to indicate if prompt changed
            })

        logger.info(f"Prepared {len(comparisons)} prompt comparisons")

        # Also include generation history
        all_generations = []
        for gen_file in prompt_files:
            match = re.search(r'gen(\d+)', gen_file.stem)
            gen_num = int(match.group(1)) if match else 0
            with gen_file.open() as f:
                gen_data = json.load(f)
                all_generations.append({
                    "generation": gen_num,
                    "prompts": gen_data,
                })

        # Load GA history data for visualizations
        ga_history = None
        ga_history_file = output_dir / "ga_history_prompts.csv"
        logger.info(f"Checking for GA history file: {ga_history_file}")
        logger.info(f"File exists: {ga_history_file.exists()}")
        if ga_history_file.exists():
            logger.info(f"Loading GA history from {ga_history_file.name}")
            try:
                ga_df = pd.read_csv(ga_history_file)
                # Remove empty rows
                ga_df = ga_df.dropna(how='all')
                
                # Extract metrics columns
                metric_cols = [col for col in ga_df.columns if col.startswith('metric::')]
                
                # Convert dataframe to records and ensure all numpy types are converted to Python types
                def convert_numpy_types(records):
                    """Convert numpy types in dict to native Python types"""
                    converted = []
                    for record in records:
                        converted_record = {}
                        for key, value in record.items():
                            if pd.isna(value):
                                converted_record[key] = None
                            elif hasattr(value, 'item'):  # numpy type
                                converted_record[key] = value.item()
                            else:
                                converted_record[key] = value
                        converted.append(converted_record)
                    return converted
                
                # Process the data for visualization
                ga_history = {
                    "raw_data": convert_numpy_types(ga_df.to_dict(orient="records")),
                    "generations": sorted([int(x) for x in ga_df['generation'].unique()]),
                    "metrics": metric_cols,
                    "summary": {
                        "total_generations": int(ga_df['generation'].max()),
                        "population_size": int(ga_df.groupby('generation')['index'].nunique().max()),
                        "total_evaluations": int(len(ga_df)),
                    },
                    # Best fitness per generation (higher is better)
                    "best_by_generation": convert_numpy_types(ga_df.loc[ga_df.groupby('generation')['scalar_fitness'].idxmax()].to_dict(orient="records")),
                    # Metrics evolution
                    "metrics_by_generation": {}
                }
                
                # Calculate statistics per generation for each metric
                for metric in metric_cols + ['scalar_fitness']:
                    metric_stats = []
                    for gen in ga_history["generations"]:
                        gen_data = ga_df[ga_df['generation'] == gen][metric]
                        metric_stats.append({
                            "generation": int(gen),
                            "mean": float(gen_data.mean()),
                            "std": float(gen_data.std()),
                            "min": float(gen_data.min()),
                            "max": float(gen_data.max()),
                            "best": float(gen_data.max()),  # Higher is better for all metrics including scalar_fitness
                        })
                    ga_history["metrics_by_generation"][metric] = metric_stats
                
                logger.info(f"Loaded GA history with {len(ga_df)} entries")
            except Exception as e:
                logger.error(f"Failed to load GA history: {e}", exc_info=True)

        result = {
            "run_id": run_id,
            "comparisons": comparisons,
            "total_prompts": len(comparisons),
            "total_generations": len(prompt_files),
            "generations": all_generations,
            "ga_history": ga_history,
        }
        
        logger.info(f"Returning prompt comparisons response with ga_history: {ga_history is not None}")
        if ga_history:
            logger.info(f"GA history summary: {ga_history.get('summary')}")
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt comparisons: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting prompt comparisons: {str(e)}")

