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
Tests for the Linear LangGraph Benchmark Agent graph construction.

Verifies that:
- The StateGraph compiles without errors
- All 7 expected nodes are present
- Edges form a strictly linear topology (no conditional edges, no branching)
- The graph topology matches the design specification
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from linear_langgraph_benchmark.graph_nodes import create_node_functions
from linear_langgraph_benchmark.graph_state import ArchitectureReviewState


@pytest.fixture(name="mock_llm")
def fixture_mock_llm():
    """Create a mock LLM that returns AIMessage responses."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(
        content="ARCHITECTURE_STYLE: microservices\nINDUSTRY: fintech\nCLOUD_PROVIDER: aws\nINITIAL_RISK: medium"))
    return llm


@pytest.fixture(name="mock_config")
def fixture_mock_config():
    """Create a mock config."""
    return SimpleNamespace()


@pytest.fixture(name="node_functions")
def fixture_node_functions(mock_llm, mock_config):
    """Create all node functions using mock LLM."""
    return create_node_functions(mock_llm, mock_config)


@pytest.fixture(name="compiled_graph")
def fixture_compiled_graph(node_functions):
    """Build and compile the full StateGraph."""
    nodes = node_functions

    graph = StateGraph(ArchitectureReviewState)

    graph.add_node("architecture_intake", nodes["architecture_intake"])
    graph.add_node("component_deep_dive", nodes["component_deep_dive"])
    graph.add_node("security_posture", nodes["security_posture"])
    graph.add_node("reliability_scaling", nodes["reliability_scaling"])
    graph.add_node("cost_efficiency", nodes["cost_efficiency"])
    graph.add_node("compliance_gaps", nodes["compliance_gaps"])
    graph.add_node("executive_verdict", nodes["executive_verdict"])

    graph.add_edge(START, "architecture_intake")
    graph.add_edge("architecture_intake", "component_deep_dive")
    graph.add_edge("component_deep_dive", "security_posture")
    graph.add_edge("security_posture", "reliability_scaling")
    graph.add_edge("reliability_scaling", "cost_efficiency")
    graph.add_edge("cost_efficiency", "compliance_gaps")
    graph.add_edge("compliance_gaps", "executive_verdict")
    graph.add_edge("executive_verdict", END)

    return graph.compile()


class TestGraphConstruction:
    """Tests for graph construction and compilation."""

    def test_graph_compiles_successfully(self, compiled_graph):
        """Graph should compile without errors."""
        assert compiled_graph is not None

    def test_all_nodes_present(self, compiled_graph):
        """All 7 registered nodes should be in the compiled graph."""
        node_names = set(compiled_graph.get_graph().nodes.keys())
        expected_nodes = {
            "architecture_intake",
            "component_deep_dive",
            "security_posture",
            "reliability_scaling",
            "cost_efficiency",
            "compliance_gaps",
            "executive_verdict",
        }
        # LangGraph adds __start__ and __end__ nodes
        for expected in expected_nodes:
            assert expected in node_names, f"Node '{expected}' not found in graph"

    def test_exactly_seven_nodes(self, compiled_graph):
        """Graph should have exactly 7 user-defined nodes (plus __start__ and __end__)."""
        node_names = set(compiled_graph.get_graph().nodes.keys())
        user_nodes = node_names - {"__start__", "__end__"}
        assert len(user_nodes) == 7

    def test_node_functions_created(self, node_functions):
        """create_node_functions should return all 7 expected functions."""
        expected_keys = {
            "architecture_intake",
            "component_deep_dive",
            "security_posture",
            "reliability_scaling",
            "cost_efficiency",
            "compliance_gaps",
            "executive_verdict",
        }
        assert set(node_functions.keys()) == expected_keys

    def test_no_conditional_edges(self, compiled_graph):
        """Graph should have no conditional edges — pure linear topology."""
        graph_data = compiled_graph.get_graph()
        for edge in graph_data.edges:
            # Conditional edges in LangGraph have a 'conditional' attribute or special structure
            # In a linear graph, all edges should be simple (source, target) tuples
            assert not getattr(edge, 'conditional', False), f"Unexpected conditional edge: {edge}"


class TestLinearTopology:
    """Tests for strictly linear graph topology."""

    def test_start_connects_to_intake(self, compiled_graph):
        """START should connect only to architecture_intake."""
        graph_data = compiled_graph.get_graph()
        start_edges = [e for e in graph_data.edges if e.source == "__start__"]
        assert len(start_edges) == 1
        assert start_edges[0].target == "architecture_intake"

    def test_verdict_connects_to_end(self, compiled_graph):
        """executive_verdict should connect only to END."""
        graph_data = compiled_graph.get_graph()
        verdict_edges = [e for e in graph_data.edges if e.source == "executive_verdict"]
        assert len(verdict_edges) == 1
        assert verdict_edges[0].target == "__end__"

    def test_linear_edge_chain(self, compiled_graph):
        """All nodes should form a single linear chain with no fan-out or fan-in."""
        graph_data = compiled_graph.get_graph()
        expected_chain = [
            ("__start__", "architecture_intake"),
            ("architecture_intake", "component_deep_dive"),
            ("component_deep_dive", "security_posture"),
            ("security_posture", "reliability_scaling"),
            ("reliability_scaling", "cost_efficiency"),
            ("cost_efficiency", "compliance_gaps"),
            ("compliance_gaps", "executive_verdict"),
            ("executive_verdict", "__end__"),
        ]

        actual_edges = [(e.source, e.target) for e in graph_data.edges]
        for expected_edge in expected_chain:
            assert expected_edge in actual_edges, f"Expected edge {expected_edge} not found"

    def test_each_node_has_exactly_one_outgoing_edge(self, compiled_graph):
        """Each node in the linear chain should have exactly one outgoing edge."""
        graph_data = compiled_graph.get_graph()
        linear_nodes = [
            "__start__",
            "architecture_intake",
            "component_deep_dive",
            "security_posture",
            "reliability_scaling",
            "cost_efficiency",
            "compliance_gaps",
            "executive_verdict",
        ]
        for node in linear_nodes:
            outgoing = [e for e in graph_data.edges if e.source == node]
            assert len(outgoing) == 1, f"Node '{node}' has {len(outgoing)} outgoing edges, expected 1"

    def test_total_edge_count(self, compiled_graph):
        """Linear graph with 7 nodes should have exactly 8 edges (including START/END)."""
        graph_data = compiled_graph.get_graph()
        assert len(graph_data.edges) == 8
