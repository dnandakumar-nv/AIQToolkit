# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
State schema for the Linear LangGraph Benchmark Agent.

Defines the ArchitectureReviewState TypedDict used by all graph nodes.
Messages are accumulated via operator.add to enable KV cache prefix reuse:
each successive LLM call sees the full conversation history as a prefix.
"""

import operator
from typing import Annotated
from typing import TypedDict


class ArchitectureReviewState(TypedDict):
    """
    State for the Enterprise Architecture Review pipeline.

    Fields:
        query: The original architecture proposal from the user.
        messages: Accumulated conversation messages (SystemMessage + HumanMessage/AIMessage pairs).
            Uses operator.add reducer so nodes return only new messages (delta).
        architecture_style: Detected architecture style (e.g., "microservices", "monolithic").
        industry: Detected industry vertical (e.g., "fintech", "healthcare").
        cloud_provider: Detected cloud provider (e.g., "aws", "gcp", "azure").
        component_findings: Deep-dive analysis of individual components.
        security_findings: Security posture assessment results.
        reliability_findings: Reliability and scaling assessment results.
        cost_findings: Cost efficiency analysis results.
        compliance_findings: Compliance gap analysis results.
        risk_level: Overall risk level ("critical", "high", "medium", "low").
        overall_score: Final architecture score (0.0 to 10.0).
        final_verdict: Executive verdict and recommendations.
    """

    query: str
    messages: Annotated[list, operator.add]
    architecture_style: str
    industry: str
    cloud_provider: str
    component_findings: str
    security_findings: str
    reliability_findings: str
    cost_findings: str
    compliance_findings: str
    risk_level: str
    overall_score: float
    final_verdict: str
