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
Linear LangGraph Benchmark Agent — NAT Registration and StateGraph Construction.

Builds a 7-node strictly linear LangGraph StateGraph for stress-testing
Dynamo's KV cache prefix reuse. Every successive LLM call uses the entire
context from all previous calls as a prefix, creating the ideal workload
for measuring prefix cache hit rates.

Graph topology (pure linear, no branching):
    START -> architecture_intake -> component_deep_dive -> security_posture ->
             reliability_scaling -> cost_efficiency -> compliance_gaps ->
             executive_verdict -> END
"""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LinearLanggraphAgentConfig(FunctionBaseConfig, name="linear_langgraph_agent"):
    """
    Configuration for the Linear LangGraph Benchmark Agent.

    This agent builds a 7-node strictly linear graph that accumulates
    messages across all nodes, generating exactly 7 LLM calls per query
    with maximal KV cache prefix reuse.
    """

    llm_name: LLMRef = Field(description="LLM to use for all graph nodes.")


@register_function(config_type=LinearLanggraphAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def linear_langgraph_agent_function(config: LinearLanggraphAgentConfig, builder: Builder):
    """
    Registers the Linear LangGraph Benchmark Agent.

    Builds and compiles a StateGraph with 7 nodes in a strictly linear
    topology. No conditional edges, no branching, no retry cycles.

    Args:
        config: Agent configuration.
        builder: NAT builder for resolving LLM references.

    Yields:
        FunctionInfo wrapping the compiled graph's invoke function.
    """
    from langgraph.graph import END
    from langgraph.graph import START
    from langgraph.graph import StateGraph

    from .graph_nodes import create_node_functions
    from .graph_state import ArchitectureReviewState

    # Resolve LLM from builder
    llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Create all node functions with LLM captured via closure
    nodes = create_node_functions(llm, config)

    # Build the StateGraph — strictly linear topology
    graph = StateGraph(ArchitectureReviewState)

    # Add all 7 nodes
    graph.add_node("architecture_intake", nodes["architecture_intake"])
    graph.add_node("component_deep_dive", nodes["component_deep_dive"])
    graph.add_node("security_posture", nodes["security_posture"])
    graph.add_node("reliability_scaling", nodes["reliability_scaling"])
    graph.add_node("cost_efficiency", nodes["cost_efficiency"])
    graph.add_node("compliance_gaps", nodes["compliance_gaps"])
    graph.add_node("executive_verdict", nodes["executive_verdict"])

    # Wire linear edges: START -> node1 -> node2 -> ... -> node7 -> END
    graph.add_edge(START, "architecture_intake")
    graph.add_edge("architecture_intake", "component_deep_dive")
    graph.add_edge("component_deep_dive", "security_posture")
    graph.add_edge("security_posture", "reliability_scaling")
    graph.add_edge("reliability_scaling", "cost_efficiency")
    graph.add_edge("cost_efficiency", "compliance_gaps")
    graph.add_edge("compliance_gaps", "executive_verdict")
    graph.add_edge("executive_verdict", END)

    # Compile the graph
    agent_executor = graph.compile()

    logger.info("Linear LangGraph agent compiled: 7 nodes, pure linear topology")

    async def run_architecture_review(query: str) -> str:
        """
        Review an enterprise architecture proposal through the full analysis pipeline.

        The graph performs 7 sequential analysis phases: intake assessment,
        component deep-dive, security posture, reliability/scaling,
        cost efficiency, compliance gaps, and executive verdict.

        Each phase sees the full conversation history from all previous phases,
        maximizing KV cache prefix reuse.

        Args:
            query: The architecture proposal to review.

        Returns:
            Executive verdict with overall score and recommendations.
        """
        initial_state = {
            "query": query,
            "messages": [],
            "architecture_style": "",
            "industry": "",
            "cloud_provider": "",
            "component_findings": "",
            "security_findings": "",
            "reliability_findings": "",
            "cost_findings": "",
            "compliance_findings": "",
            "risk_level": "",
            "overall_score": 0.0,
            "final_verdict": "",
        }

        result = await agent_executor.ainvoke(initial_state)

        return result.get("final_verdict", "No verdict generated.")

    yield FunctionInfo.from_fn(
        run_architecture_review,
        description="Linear LangGraph agent for enterprise architecture review with 7 sequential "
        "analysis phases. Generates exactly 7 LLM calls per query with maximal KV cache prefix reuse.",
    )
