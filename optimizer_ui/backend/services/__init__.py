# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared services module.
"""

from optimizer_ui.backend.services.optimization_service import OptimizationService

# Create a single shared instance that all routes can use
_shared_optimization_service = None


def get_optimization_service() -> OptimizationService:
    """Get the shared optimization service instance."""
    global _shared_optimization_service
    if _shared_optimization_service is None:
        _shared_optimization_service = OptimizationService()
    return _shared_optimization_service
