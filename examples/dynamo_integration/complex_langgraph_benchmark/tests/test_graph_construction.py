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
Tests for the Complex LangGraph Benchmark Agent graph construction.

Verifies that:
- The StateGraph compiles without errors
- All expected nodes are present
- Edges and conditional edges are correctly wired
- The graph topology matches the design specification
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph

from complex_langgraph_benchmark.graph_nodes import create_node_functions
from complex_langgraph_benchmark.graph_state import IncidentState


@pytest.fixture(name="mock_llm")
def fixture_mock_llm():
    """Create a mock LLM that returns structured responses."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="SEVERITY: standard\nCATEGORY: mixed\nREASONING: test"))
    return llm


@pytest.fixture(name="mock_config")
def fixture_mock_config():
    """Create a mock config with default values."""
    return SimpleNamespace(
        max_quality_retries=2,
        max_synthesis_retries=2,
        quality_threshold=0.7,
        decision_only=True,
    )


@pytest.fixture(name="node_functions")
def fixture_node_functions(mock_llm, mock_config):
    """Create all node functions using mock LLM."""
    return create_node_functions(mock_llm, mock_config)


@pytest.fixture(name="compiled_graph")
def fixture_compiled_graph(node_functions):
    """Build and compile the full StateGraph."""
    nodes = node_functions

    graph = StateGraph(IncidentState)

    graph.add_node("classify_incident", nodes["classify_incident"])
    graph.add_node("analyze_branch", nodes["analyze_branch"])
    graph.add_node("aggregate_findings", nodes["aggregate_findings"])
    graph.add_node("assess_quality", nodes["assess_quality"])
    graph.add_node("synthesize_report", nodes["synthesize_report"])
    graph.add_node("critique_report", nodes["critique_report"])
    graph.add_node("extract_actions", nodes["extract_actions"])
    graph.add_node("risk_assessment", nodes["risk_assessment"])
    graph.add_node("final_summary", nodes["final_summary"])

    graph.add_edge(START, "classify_incident")
    graph.add_conditional_edges("classify_incident", nodes["fan_out_router"])
    graph.add_edge("analyze_branch", "aggregate_findings")
    graph.add_edge("aggregate_findings", "assess_quality")
    graph.add_conditional_edges("assess_quality", nodes["quality_gate"])
    graph.add_edge("synthesize_report", "critique_report")
    graph.add_conditional_edges("critique_report", nodes["critique_gate"], {
        "synthesize_report": "synthesize_report",
        "extract_actions": "extract_actions",
    })
    graph.add_edge("extract_actions", "risk_assessment")
    graph.add_edge("risk_assessment", "final_summary")
    graph.add_edge("final_summary", END)

    return graph.compile()


class TestGraphConstruction:
    """Tests for graph construction and compilation."""

    def test_graph_compiles_successfully(self, compiled_graph):
        """Graph should compile without errors."""
        assert compiled_graph is not None

    def test_all_nodes_present(self, compiled_graph):
        """All 9 registered nodes should be in the compiled graph."""
        node_names = set(compiled_graph.get_graph().nodes.keys())
        expected_nodes = {
            "classify_incident",
            "analyze_branch",
            "aggregate_findings",
            "assess_quality",
            "synthesize_report",
            "critique_report",
            "extract_actions",
            "risk_assessment",
            "final_summary",
        }
        # LangGraph adds __start__ and __end__ nodes
        for expected in expected_nodes:
            assert expected in node_names, f"Node '{expected}' not found in graph"

    def test_node_functions_created(self, node_functions):
        """create_node_functions should return all expected functions."""
        expected_keys = {
            "classify_incident",
            "fan_out_router",
            "analyze_branch",
            "aggregate_findings",
            "assess_quality",
            "synthesize_report",
            "critique_report",
            "extract_actions",
            "risk_assessment",
            "final_summary",
            "quality_gate",
            "critique_gate",
        }
        assert set(node_functions.keys()) == expected_keys


class TestConditionalEdges:
    """Tests for conditional edge functions."""

    def test_fan_out_router_returns_sends(self, node_functions):
        """fan_out_router should return Send objects for 3 parallel branches."""
        router_fn = node_functions["fan_out_router"]
        state = {
            "query": "test",
            "incident_id": "test-001",
            "severity": "critical",
            "category": "security",
            "classification_reasoning": "test reasoning",
        }
        result = router_fn(state)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_quality_gate_retries_on_low_score(self, node_functions):
        """quality_gate should return Send objects when score is below threshold."""
        gate_fn = node_functions["quality_gate"]
        state = {
            "quality_score": 0.3,
            "quality_attempts": 0,
            "query": "test",
            "severity": "standard",
            "category": "mixed",
            "classification_reasoning": "test",
        }
        result = gate_fn(state)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_quality_gate_proceeds_on_high_score(self, node_functions):
        """quality_gate should proceed when score is above threshold."""
        gate_fn = node_functions["quality_gate"]
        state = {"quality_score": 0.9, "quality_attempts": 0}
        assert gate_fn(state) == "synthesize_report"

    def test_quality_gate_proceeds_on_max_retries(self, node_functions):
        """quality_gate should proceed when max retries reached even with low score."""
        gate_fn = node_functions["quality_gate"]
        state = {"quality_score": 0.3, "quality_attempts": 2}
        assert gate_fn(state) == "synthesize_report"

    def test_critique_gate_revises_on_revise_feedback(self, node_functions):
        """critique_gate should retry synthesis when REVISE is in feedback."""
        gate_fn = node_functions["critique_gate"]
        state = {"critique_feedback": "VERDICT: REVISE\nFEEDBACK: needs more detail", "synthesis_attempts": 1}
        assert gate_fn(state) == "synthesize_report"

    def test_critique_gate_accepts_on_accept_feedback(self, node_functions):
        """critique_gate should proceed when feedback contains ACCEPT."""
        gate_fn = node_functions["critique_gate"]
        state = {"critique_feedback": "VERDICT: ACCEPT\nFEEDBACK: looks good", "synthesis_attempts": 1}
        assert gate_fn(state) == "extract_actions"

    def test_critique_gate_proceeds_on_max_retries(self, node_functions):
        """critique_gate should proceed when max retries reached."""
        gate_fn = node_functions["critique_gate"]
        state = {"critique_feedback": "VERDICT: REVISE\nFEEDBACK: needs work", "synthesis_attempts": 2}
        assert gate_fn(state) == "extract_actions"


class TestFanOutRouter:
    """Tests for the fan-out router node."""

    def test_fan_out_returns_three_sends(self, node_functions):
        """fan_out_router should dispatch 3 Send objects."""
        router = node_functions["fan_out_router"]
        state = {
            "query": "test incident",
            "incident_id": "test-001",
            "severity": "critical",
            "category": "security",
            "classification_reasoning": "test reasoning",
        }
        sends = router(state)
        assert len(sends) == 3

    def test_fan_out_sends_correct_branches(self, node_functions):
        """fan_out_router should create sends for security, performance, and infrastructure."""
        router = node_functions["fan_out_router"]
        state = {
            "query": "test incident",
            "incident_id": "test-001",
            "severity": "critical",
            "category": "security",
            "classification_reasoning": "test reasoning",
        }
        sends = router(state)
        branches = [s.arg["branch"] for s in sends]
        assert set(branches) == {"security", "performance", "infrastructure"}
