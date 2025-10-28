# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Async wrapper for the optimizer to properly handle progress monitoring.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any
import concurrent.futures

from nat.data_models.optimizer import OptimizerRunConfig
from nat.profiler.parameter_optimization.optimizer_runtime import optimize_config

logger = logging.getLogger(__name__)


class AsyncOptimizer:
    """Wrapper to make optimization truly async with progress monitoring."""
    
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    async def run_optimization_with_progress(
        self,
        opt_run_config: OptimizerRunConfig,
        progress_callback: Any = None
    ):
        """
        Run optimization in a thread and monitor progress.
        
        This ensures the async function doesn't return until optimization is complete.
        """
        loop = asyncio.get_event_loop()
        
        # Create a future to track completion
        optimization_future = loop.run_in_executor(
            self.executor,
            self._run_sync_optimization,
            opt_run_config
        )
        
        # Create progress monitoring task
        output_path = None
        try:
            # Get output path from config
            from nat.runtime.loader import load_config
            config_obj = load_config(config_file=opt_run_config.config_file)
            output_path = config_obj.optimizer.output_path
        except Exception as e:
            logger.warning(f"Could not get output path for monitoring: {e}")
        
        monitor_task = None
        if output_path and progress_callback:
            monitor_task = asyncio.create_task(
                self._monitor_progress(output_path, progress_callback)
            )
        
        try:
            # Wait for optimization to complete
            result = await optimization_future
            return result
        finally:
            # Cancel monitoring when done
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
    
    def _run_sync_optimization(self, opt_run_config: OptimizerRunConfig):
        """Run the synchronous optimization in a thread."""
        # Force matplotlib to use non-GUI backend to avoid thread issues
        import matplotlib
        matplotlib.use('Agg')
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the async optimization function
            return loop.run_until_complete(optimize_config(opt_run_config))
        finally:
            loop.close()
    
    async def _monitor_progress(self, output_path: Path, progress_callback):
        """Monitor optimization progress by checking output files."""
        last_trial_count = 0
        
        while True:
            await asyncio.sleep(2)  # Check every 2 seconds
            
            # Look for trial files
            trial_files = list(output_path.glob("config_numeric_trial_*.yml"))
            current_trial_count = len(trial_files)
            
            if current_trial_count > last_trial_count:
                last_trial_count = current_trial_count
                # Call the progress callback
                if progress_callback:
                    await progress_callback(current_trial_count)
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
