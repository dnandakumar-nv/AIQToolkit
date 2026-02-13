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
Complex LangGraph Benchmark Agent — NAT Registration and StateGraph Construction.

Builds a 12-node LangGraph StateGraph with parallel branches (Send fan-out),
conditional edges, and cycles for stress-testing Dynamo inference optimizations.

Graph topology:
    START → classify_incident → fan_out_router → [security, perf, infra] (parallel)
    → aggregate_findings → assess_quality → (quality_gate) → synthesize_report
    → critique_report → (critique_gate) → extract_actions → risk_assessment
    → final_summary → END
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


class ComplexLanggraphAgentConfig(FunctionBaseConfig, name="complex_langgraph_agent"):
    """
    Configuration for the Complex LangGraph Benchmark Agent.

    This agent builds a 12-node graph with 3 parallel branches, 3 conditional
    edges, and 2 cycles, generating 9-18 LLM calls per query.
    """

    llm_name: LLMRef = Field(description="LLM to use for all graph nodes.")
    decision_only: bool = Field(
        default=True,
        description="If True, tools capture intents without execution.",
    )
    max_quality_retries: int = Field(
        default=2,
        description="Maximum number of quality gate retry attempts.",
    )
    max_synthesis_retries: int = Field(
        default=2,
        description="Maximum number of synthesis/critique retry attempts.",
    )
    quality_threshold: float = Field(
        default=0.7,
        description="Quality score threshold (0.0-1.0) below which analysis is retried.",
    )


@register_function(config_type=ComplexLanggraphAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def complex_langgraph_agent_function(config: ComplexLanggraphAgentConfig, builder: Builder):
    """
    Registers the Complex LangGraph Benchmark Agent.

    Builds and compiles a StateGraph with 12 nodes, conditional edges, parallel
    branches via Send(), and cycles for quality/critique retry loops.

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
    from .graph_state import IncidentState

    # Resolve LLM from builder
    llm = await builder.get_llm(llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Create all node functions with LLM captured via closure
    nodes = create_node_functions(llm, config)

    # Build the StateGraph
    graph = StateGraph(IncidentState)

    # Add all nodes (fan_out_router is a conditional edge function, not a node)
    graph.add_node("classify_incident", nodes["classify_incident"])
    graph.add_node("analyze_branch", nodes["analyze_branch"])
    graph.add_node("aggregate_findings", nodes["aggregate_findings"])
    graph.add_node("assess_quality", nodes["assess_quality"])
    graph.add_node("synthesize_report", nodes["synthesize_report"])
    graph.add_node("critique_report", nodes["critique_report"])
    graph.add_node("extract_actions", nodes["extract_actions"])
    graph.add_node("risk_assessment", nodes["risk_assessment"])
    graph.add_node("final_summary", nodes["final_summary"])

    # Edge: START → classify_incident
    graph.add_edge(START, "classify_incident")

    # Conditional edge: classify_incident → fan_out_router (dispatches Send() to 3 parallel branches)
    # fan_out_router returns list[Send] which LangGraph uses to spawn parallel analyze_branch nodes
    graph.add_conditional_edges("classify_incident", nodes["fan_out_router"])

    # Edge: analyze_branch → aggregate_findings (fan-in after parallel)
    graph.add_edge("analyze_branch", "aggregate_findings")

    # Edge: aggregate_findings → assess_quality
    graph.add_edge("aggregate_findings", "assess_quality")

    # Conditional edge: assess_quality → quality_gate
    # CYCLE 1: quality_gate can loop back (re-dispatch Send()) or proceed to synthesize
    graph.add_conditional_edges("assess_quality", nodes["quality_gate"])

    # Edge: synthesize_report → critique_report
    graph.add_edge("synthesize_report", "critique_report")

    # Conditional edge: critique_report → critique_gate
    # CYCLE 2: critique_gate can loop back to synthesize_report if revision needed
    graph.add_conditional_edges("critique_report", nodes["critique_gate"], {
        "synthesize_report": "synthesize_report",
        "extract_actions": "extract_actions",
    })

    # Edge: extract_actions → risk_assessment
    graph.add_edge("extract_actions", "risk_assessment")

    # Edge: risk_assessment → final_summary
    graph.add_edge("risk_assessment", "final_summary")

    # Edge: final_summary → END
    graph.add_edge("final_summary", END)

    # Compile the graph
    agent_executor = graph.compile()

    logger.info("Complex LangGraph agent compiled: 10 nodes, 3 parallel branches, 2 cycles")

    async def run_incident_response(query: str) -> str:
        """
        Process an IT incident through the full analysis pipeline.

        The graph classifies the incident, runs 3 parallel analyses (security,
        performance, infrastructure), assesses quality with retry loop,
        synthesizes a report with critique loop, extracts actions, assesses
        risk, and generates a final comprehensive summary.

        Args:
            query: The incident description to analyze.

        Returns:
            Comprehensive incident response summary.
        """
        initial_state = {
            "query": query,
            "incident_id": "",
            "severity": "",
            "category": "",
            "classification_reasoning": "",
            "analysis_results": [],
            "aggregated_findings": "",
            "quality_score": 0.0,
            "quality_feedback": "",
            "quality_attempts": 0,
            "draft_report": "",
            "critique_feedback": "",
            "synthesis_attempts": 0,
            "tool_calls_made": [],
            "recommended_actions": [],
            "risk_level": "",
            "risk_reasoning": "",
            "final_response": "",
            "messages": [],
        }

        result = await agent_executor.ainvoke(initial_state)

        return result.get("final_response", "No response generated.")

    yield FunctionInfo.from_fn(
        run_incident_response,
        description="Complex LangGraph agent for IT incident response with parallel analysis, "
        "quality gates, and critique cycles. Generates 9-18 LLM calls per query.",
    )
