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
Graph node functions for the Linear LangGraph Benchmark Agent.

Contains 7 async node functions for the Enterprise Architecture Review pipeline.
Each node appends to a shared messages list and calls llm.ainvoke(messages),
creating a strict prefix-extension pattern for KV cache reuse.

Graph topology (pure linear, no branching):
    START -> architecture_intake -> component_deep_dive -> security_posture ->
             reliability_scaling -> cost_efficiency -> compliance_gaps ->
             executive_verdict -> END
"""

import logging
import re

from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from nat.profiler.decorators.latency import latency_sensitive

from . import prompts
from .graph_state import ArchitectureReviewState

logger = logging.getLogger(__name__)


def create_node_functions(llm, config):
    """
    Create all graph node functions with the LLM instance captured via closure.

    Each node follows the messages-list pattern for KV cache prefix reuse:
    1. Read the full conversation from state["messages"]
    2. Append a new HumanMessage with the step prompt
    3. Call llm.ainvoke(messages) so the LLM sees the full prefix
    4. Return {"messages": [human_msg, ai_response], ...} as delta

    Args:
        llm: LangChain-wrapped LLM instance from the NAT builder.
        config: Agent configuration (unused for linear benchmark but kept for pattern consistency).

    Returns:
        Dictionary mapping node names to async node functions.
    """

    # =========================================================================
    # Node 1: architecture_intake — First LLM call with system prompt + proposal
    # =========================================================================
    @latency_sensitive("low")
    async def architecture_intake(state: ArchitectureReviewState) -> dict:
        """Perform initial intake assessment of the architecture proposal."""
        proposal = state["query"]

        # Build initial messages: system prompt + intake prompt with proposal
        system_msg = SystemMessage(content=prompts.SYSTEM_PROMPT)
        human_msg = HumanMessage(content=prompts.INTAKE_PROMPT.format(proposal=proposal))
        messages = [system_msg, human_msg]

        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        # Parse structured fields from response
        architecture_style = "unknown"
        industry = "unknown"
        cloud_provider = "unknown"

        style_match = re.search(r'ARCHITECTURE_STYLE:\s*(\S+)', content, re.IGNORECASE)
        if style_match:
            architecture_style = style_match.group(1).lower().strip()

        industry_match = re.search(r'INDUSTRY:\s*(\S+)', content, re.IGNORECASE)
        if industry_match:
            industry = industry_match.group(1).lower().strip()

        cloud_match = re.search(r'CLOUD_PROVIDER:\s*(\S+)', content, re.IGNORECASE)
        if cloud_match:
            cloud_provider = cloud_match.group(1).lower().strip()

        logger.info("Architecture intake: style=%s, industry=%s, cloud=%s",
                    architecture_style,
                    industry,
                    cloud_provider)

        return {
            "messages": [system_msg, human_msg, response],
            "architecture_style": architecture_style,
            "industry": industry,
            "cloud_provider": cloud_provider,
        }

    # =========================================================================
    # Node 2: component_deep_dive — Analyze individual components
    # =========================================================================
    @latency_sensitive("low")
    async def component_deep_dive(state: ArchitectureReviewState) -> dict:
        """Deep-dive analysis of individual architecture components."""
        messages = list(state["messages"])
        human_msg = HumanMessage(content=prompts.COMPONENT_DEEP_DIVE_PROMPT)
        messages.append(human_msg)

        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        logger.info("Component deep-dive completed (%d chars)", len(content))

        return {
            "messages": [human_msg, response],
            "component_findings": content,
        }

    # =========================================================================
    # Node 3: security_posture — Evaluate security against frameworks
    # =========================================================================
    @latency_sensitive("low")
    async def security_posture(state: ArchitectureReviewState) -> dict:
        """Evaluate security posture against OWASP, CIS, and NIST frameworks."""
        messages = list(state["messages"])
        human_msg = HumanMessage(content=prompts.SECURITY_POSTURE_PROMPT)
        messages.append(human_msg)

        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        logger.info("Security posture assessment completed (%d chars)", len(content))

        return {
            "messages": [human_msg, response],
            "security_findings": content,
        }

    # =========================================================================
    # Node 4: reliability_scaling — Assess reliability and scaling
    # =========================================================================
    @latency_sensitive("low")
    async def reliability_scaling(state: ArchitectureReviewState) -> dict:
        """Assess reliability characteristics and scaling strategies."""
        messages = list(state["messages"])
        human_msg = HumanMessage(content=prompts.RELIABILITY_SCALING_PROMPT)
        messages.append(human_msg)

        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        logger.info("Reliability/scaling assessment completed (%d chars)", len(content))

        return {
            "messages": [human_msg, response],
            "reliability_findings": content,
        }

    # =========================================================================
    # Node 5: cost_efficiency — Evaluate cost optimization
    # =========================================================================
    @latency_sensitive("low")
    async def cost_efficiency(state: ArchitectureReviewState) -> dict:
        """Evaluate cost efficiency and resource optimization."""
        messages = list(state["messages"])
        human_msg = HumanMessage(content=prompts.COST_EFFICIENCY_PROMPT)
        messages.append(human_msg)

        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        logger.info("Cost efficiency assessment completed (%d chars)", len(content))

        return {
            "messages": [human_msg, response],
            "cost_findings": content,
        }

    # =========================================================================
    # Node 6: compliance_gaps — Identify compliance gaps
    # =========================================================================
    @latency_sensitive("low")
    async def compliance_gaps(state: ArchitectureReviewState) -> dict:
        """Identify compliance gaps against applicable regulatory standards."""
        messages = list(state["messages"])
        human_msg = HumanMessage(content=prompts.COMPLIANCE_GAPS_PROMPT)
        messages.append(human_msg)

        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        logger.info("Compliance gap analysis completed (%d chars)", len(content))

        return {
            "messages": [human_msg, response],
            "compliance_findings": content,
        }

    # =========================================================================
    # Node 7: executive_verdict — Final synthesis and scoring
    # =========================================================================
    @latency_sensitive("low")
    async def executive_verdict(state: ArchitectureReviewState) -> dict:
        """Synthesize all analysis phases into a final executive verdict."""
        messages = list(state["messages"])
        human_msg = HumanMessage(content=prompts.EXECUTIVE_VERDICT_PROMPT)
        messages.append(human_msg)

        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, 'content') else str(response)

        # Parse risk level and overall score
        risk_level = "medium"
        overall_score = 5.0

        level_match = re.search(r'RISK_LEVEL:\s*(\w+)', content, re.IGNORECASE)
        if level_match:
            risk_level = level_match.group(1).lower()
            if risk_level not in ("critical", "low", "medium", "low"):
                risk_level = "medium"

        score_match = re.search(r'OVERALL_SCORE:\s*([\d.]+)', content, re.IGNORECASE)
        if score_match:
            try:
                overall_score = float(score_match.group(1))
                overall_score = max(0.0, min(10.0, overall_score))
            except ValueError:
                overall_score = 5.0

        logger.info("Executive verdict: risk=%s, score=%.1f", risk_level, overall_score)

        return {
            "messages": [human_msg, response],
            "risk_level": risk_level,
            "overall_score": overall_score,
            "final_verdict": content,
        }

    return {
        "architecture_intake": architecture_intake,
        "component_deep_dive": component_deep_dive,
        "security_posture": security_posture,
        "reliability_scaling": reliability_scaling,
        "cost_efficiency": cost_efficiency,
        "compliance_gaps": compliance_gaps,
        "executive_verdict": executive_verdict,
    }
