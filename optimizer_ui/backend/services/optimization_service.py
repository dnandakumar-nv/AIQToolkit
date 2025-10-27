# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Service for managing optimization runs with progress tracking.
"""

import asyncio
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from nat.data_models.config import Config
from nat.data_models.optimizer import OptimizerRunConfig
from nat.profiler.parameter_optimization.optimizer_runtime import optimize_config

logger = logging.getLogger(__name__)


class OptimizationService:
    """
    Service to manage optimization runs with progress tracking and WebSocket updates.
    """

    def __init__(self):
        self.runs: dict[str, dict[str, Any]] = {}
        self.subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self.tasks: dict[str, asyncio.Task] = {}

    async def start_optimization(
        self,
        config_path: str,
        dataset_path: str | None = None,
        result_json_path: str = "$",
        endpoint: str | None = None,
        endpoint_timeout: int = 300,
    ) -> str:
        """
        Start a new optimization run using a config file path.

        Args:
            config_path: Absolute path to the configuration file

        Returns:
            run_id: Unique identifier for this optimization run
        """
        run_id = str(uuid.uuid4())[:8]

        # Validate config file exists
        config_file_path = Path(config_path)
        if not config_file_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        # Load config to validate and get optimizer settings
        try:
            from nat.runtime.loader import load_config
            config_obj = load_config(config_file=config_file_path)
        except Exception as e:
            logger.error(f"Invalid config: {e}")
            raise ValueError(f"Invalid configuration: {str(e)}")

        # Ensure output path is set
        if config_obj.optimizer.output_path is None:
            config_obj.optimizer.output_path = Path(f"optimizer_results/{run_id}")
            # We need to update the config file with the output path
            import yaml
            with config_file_path.open("r") as f:
                config_dict = yaml.safe_load(f)
            if "optimizer" not in config_dict:
                config_dict["optimizer"] = {}
            config_dict["optimizer"]["output_path"] = str(config_obj.optimizer.output_path)
            with config_file_path.open("w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # Create OptimizerRunConfig with file path instead of config object
        opt_run_config = OptimizerRunConfig(
            config_file=config_file_path,  # Pass the path directly
            dataset=dataset_path,
            result_json_path=result_json_path,
            endpoint=endpoint,
            endpoint_timeout=endpoint_timeout,
        )

        # Initialize run status
        self.runs[run_id] = {
            "run_id": run_id,
            "status": "initializing",
            "progress": 0.0,
            "current_trial": None,
            "total_trials": None,
            "message": "Starting optimization...",
            "result_path": str(config_obj.optimizer.output_path),
            "config_path": config_path,
        }

        # Start optimization task
        task = asyncio.create_task(self._run_optimization(run_id, opt_run_config))
        self.tasks[run_id] = task

        # Send initial update
        await self._send_update(run_id, self.runs[run_id])

        return run_id

    async def _run_optimization(self, run_id: str, opt_run_config: OptimizerRunConfig):
        """
        Execute the optimization in the background.
        """
        try:
            # Update status to running
            self.runs[run_id]["status"] = "running"
            self.runs[run_id]["message"] = "Optimization in progress..."
            await self._send_update(run_id, self.runs[run_id])

            # Load config to extract total trials
            from nat.runtime.loader import load_config
            config_obj = load_config(config_file=opt_run_config.config_file)
            if config_obj.optimizer.numeric.enabled:
                self.runs[run_id]["total_trials"] = config_obj.optimizer.numeric.n_trials

            # Run optimization with progress callback
            logger.info(f"Starting optimization for run {run_id}")

            # Create a wrapper to track progress
            # Note: The actual optimizer doesn't have a built-in progress callback,
            # so we'll use asyncio to periodically check and estimate progress

            # Start progress monitoring task
            monitor_task = asyncio.create_task(
                self._monitor_progress(run_id, opt_run_config.config_file.optimizer.output_path)
            )

            # Run optimization
            result = await optimize_config(opt_run_config)

            # Cancel monitor task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Update status to completed
            self.runs[run_id]["status"] = "completed"
            self.runs[run_id]["progress"] = 100.0
            self.runs[run_id]["message"] = "Optimization completed successfully!"
            await self._send_update(run_id, self.runs[run_id])

            logger.info(f"Optimization {run_id} completed successfully")

        except asyncio.CancelledError:
            self.runs[run_id]["status"] = "stopped"
            self.runs[run_id]["message"] = "Optimization stopped by user"
            await self._send_update(run_id, self.runs[run_id])
            logger.info(f"Optimization {run_id} was cancelled")

        except Exception as e:
            self.runs[run_id]["status"] = "failed"
            self.runs[run_id]["message"] = f"Optimization failed: {str(e)}"
            await self._send_update(run_id, self.runs[run_id])
            logger.error(f"Optimization {run_id} failed: {e}", exc_info=True)

    async def _monitor_progress(self, run_id: str, output_path: Path | None):
        """
        Monitor optimization progress by checking the output directory.
        """
        if not output_path:
            return

        last_trial_count = 0

        try:
            # Ensure output path exists
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            while True:
                await asyncio.sleep(5)  # Check every 5 seconds

                # Look for intermediate config files to estimate progress
                trial_files = list(output_path.glob("config_numeric_trial_*.yml"))
                current_trial_count = len(trial_files)

                if current_trial_count > last_trial_count:
                    last_trial_count = current_trial_count
                    self.runs[run_id]["current_trial"] = current_trial_count

                    # Calculate progress
                    total_trials = self.runs[run_id].get("total_trials")
                    if total_trials:
                        progress = (current_trial_count / total_trials) * 100
                        self.runs[run_id]["progress"] = min(progress, 99.0)  # Cap at 99% until complete
                        self.runs[run_id]["message"] = f"Running trial {current_trial_count}/{total_trials}..."
                    else:
                        self.runs[run_id]["progress"] = min(current_trial_count * 5, 99.0)
                        self.runs[run_id]["message"] = f"Running trial {current_trial_count}..."

                    await self._send_update(run_id, self.runs[run_id])

        except asyncio.CancelledError:
            pass

    async def _send_update(self, run_id: str, data: dict[str, Any]):
        """
        Send update to all subscribers for this run.
        """
        if run_id in self.subscribers:
            for queue in self.subscribers[run_id]:
                try:
                    await queue.put(data)
                except Exception as e:
                    logger.error(f"Error sending update to subscriber: {e}")

    def subscribe(self, run_id: str) -> asyncio.Queue:
        """
        Subscribe to updates for a specific run.

        Returns:
            Queue that will receive updates
        """
        queue = asyncio.Queue()
        self.subscribers[run_id].append(queue)

        # Send current status immediately if available
        if run_id in self.runs:
            try:
                queue.put_nowait(self.runs[run_id])
            except asyncio.QueueFull:
                pass

        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue):
        """
        Unsubscribe from updates for a specific run.
        """
        if run_id in self.subscribers:
            try:
                self.subscribers[run_id].remove(queue)
            except ValueError:
                pass

    def get_status(self, run_id: str) -> dict[str, Any] | None:
        """
        Get the current status of a run.
        """
        return self.runs.get(run_id)

    async def stop_optimization(self, run_id: str) -> bool:
        """
        Stop a running optimization.

        Returns:
            True if stopped successfully, False if run not found
        """
        if run_id not in self.tasks:
            return False

        task = self.tasks[run_id]
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        return True

    def list_runs(self) -> list[dict[str, Any]]:
        """
        List all optimization runs.
        """
        return list(self.runs.values())

    async def cleanup(self):
        """
        Clean up all running tasks.
        """
        for run_id, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
