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
Tests for individual graph node functions.

Verifies that each node function:
- Correctly appends to the messages list (prefix-extension pattern)
- Returns delta messages only (not the full conversation)
- Produces correct state updates from mock LLM responses
- Uses AIMessage objects (not MagicMock) for message accumulation
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from linear_langgraph_benchmark.graph_nodes import create_node_functions


@pytest.fixture(name="mock_config")
def fixture_mock_config():
    """Create a mock config."""
    return SimpleNamespace()


def _make_mock_llm(response_text: str):
    """Create a mock LLM returning a specific AIMessage response."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=response_text))
    return llm


class TestArchitectureIntake:
    """Tests for the architecture_intake node."""

    async def test_parses_architecture_fields(self, mock_config):
        """architecture_intake should parse style, industry, and cloud from LLM response."""
        llm = _make_mock_llm(
            "ARCHITECTURE_STYLE: microservices\nINDUSTRY: fintech\nCLOUD_PROVIDER: aws\nINITIAL_RISK: high\n"
            "This is a well-structured microservices architecture for financial services.")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "A microservices architecture for payment processing on AWS"}
        result = await nodes["architecture_intake"](state)

        assert result["architecture_style"] == "microservices"
        assert result["industry"] == "fintech"
        assert result["cloud_provider"] == "aws"

    async def test_returns_system_human_ai_messages(self, mock_config):
        """architecture_intake should return [SystemMessage, HumanMessage, AIMessage] as delta."""
        llm = _make_mock_llm("ARCHITECTURE_STYLE: monolithic\nINDUSTRY: healthcare\nCLOUD_PROVIDER: gcp")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "A monolithic healthcare app on GCP"}
        result = await nodes["architecture_intake"](state)

        messages = result["messages"]
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert isinstance(messages[2], AIMessage)

    async def test_defaults_on_unparseable_response(self, mock_config):
        """architecture_intake should use defaults when LLM response is unstructured."""
        llm = _make_mock_llm("This architecture looks interesting and has some good elements.")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "Some architecture"}
        result = await nodes["architecture_intake"](state)

        assert result["architecture_style"] == "unknown"
        assert result["industry"] == "unknown"
        assert result["cloud_provider"] == "unknown"


class TestComponentDeepDive:
    """Tests for the component_deep_dive node."""

    async def test_appends_to_messages(self, mock_config):
        """component_deep_dive should read existing messages and append new ones."""
        llm = _make_mock_llm("Component analysis: API gateway is well-designed.\nCOMPONENT_RISK: low")
        nodes = create_node_functions(llm, mock_config)

        existing_messages = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="intake prompt"),
            AIMessage(content="intake response"),
        ]

        state = {"query": "test", "messages": existing_messages}
        result = await nodes["component_deep_dive"](state)

        # Should return delta only: [HumanMessage, AIMessage]
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], HumanMessage)
        assert isinstance(result["messages"][1], AIMessage)

    async def test_invokes_llm_with_full_context(self, mock_config):
        """component_deep_dive should pass all existing messages plus new prompt to LLM."""
        llm = _make_mock_llm("COMPONENT_RISK: medium")
        nodes = create_node_functions(llm, mock_config)

        existing_messages = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="intake prompt"),
            AIMessage(content="intake response"),
        ]

        state = {"query": "test", "messages": existing_messages}
        await nodes["component_deep_dive"](state)

        # Verify LLM was called with 4 messages (3 existing + 1 new HumanMessage)
        call_args = llm.ainvoke.call_args[0][0]
        assert len(call_args) == 4
        assert isinstance(call_args[3], HumanMessage)

    async def test_stores_component_findings(self, mock_config):
        """component_deep_dive should store the full response as component_findings."""
        response_text = "The API gateway uses proper rate limiting.\nCOMPONENT_RISK: low"
        llm = _make_mock_llm(response_text)
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "test", "messages": [SystemMessage(content="sys")]}
        result = await nodes["component_deep_dive"](state)

        assert result["component_findings"] == response_text


class TestSecurityPosture:
    """Tests for the security_posture node."""

    async def test_returns_delta_messages(self, mock_config):
        """security_posture should return only the new HumanMessage and AIMessage."""
        llm = _make_mock_llm("SECURITY_RISK: high\nMultiple OWASP concerns identified.")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "test", "messages": [SystemMessage(content="sys"), HumanMessage(content="q")]}
        result = await nodes["security_posture"](state)

        assert len(result["messages"]) == 2
        assert "security_findings" in result


class TestReliabilityScaling:
    """Tests for the reliability_scaling node."""

    async def test_returns_delta_messages(self, mock_config):
        """reliability_scaling should return only delta messages."""
        llm = _make_mock_llm("RELIABILITY_RISK: medium\nSingle point of failure in database layer.")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "test", "messages": [SystemMessage(content="sys")]}
        result = await nodes["reliability_scaling"](state)

        assert len(result["messages"]) == 2
        assert "reliability_findings" in result


class TestCostEfficiency:
    """Tests for the cost_efficiency node."""

    async def test_returns_delta_messages(self, mock_config):
        """cost_efficiency should return only delta messages."""
        llm = _make_mock_llm("COST_RISK: low\nGood use of reserved instances.")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "test", "messages": [SystemMessage(content="sys")]}
        result = await nodes["cost_efficiency"](state)

        assert len(result["messages"]) == 2
        assert "cost_findings" in result


class TestComplianceGaps:
    """Tests for the compliance_gaps node."""

    async def test_returns_delta_messages(self, mock_config):
        """compliance_gaps should return only delta messages."""
        llm = _make_mock_llm("COMPLIANCE_RISK: high\nGDPR data residency requirements not met.")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "test", "messages": [SystemMessage(content="sys")]}
        result = await nodes["compliance_gaps"](state)

        assert len(result["messages"]) == 2
        assert "compliance_findings" in result


class TestExecutiveVerdict:
    """Tests for the executive_verdict node."""

    async def test_parses_risk_and_score(self, mock_config):
        """executive_verdict should parse risk level and overall score from LLM response."""
        llm = _make_mock_llm(
            "RISK_LEVEL: high\nOVERALL_SCORE: 6.5\nVERDICT: conditional_approve\n"
            "The architecture requires significant security improvements before production deployment.")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "test", "messages": [SystemMessage(content="sys")]}
        result = await nodes["executive_verdict"](state)

        assert result["risk_level"] == "high"
        assert result["overall_score"] == 6.5
        assert "final_verdict" in result

    async def test_clamps_score_to_valid_range(self, mock_config):
        """executive_verdict should clamp score between 0.0 and 10.0."""
        llm = _make_mock_llm("RISK_LEVEL: low\nOVERALL_SCORE: 15.0\nVERDICT: approve")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "test", "messages": [SystemMessage(content="sys")]}
        result = await nodes["executive_verdict"](state)

        assert result["overall_score"] == 10.0

    async def test_defaults_on_unparseable_response(self, mock_config):
        """executive_verdict should use defaults when response lacks structured fields."""
        llm = _make_mock_llm("Overall this architecture is acceptable.")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "test", "messages": [SystemMessage(content="sys")]}
        result = await nodes["executive_verdict"](state)

        assert result["risk_level"] == "medium"
        assert result["overall_score"] == 5.0

    async def test_returns_delta_messages(self, mock_config):
        """executive_verdict should return only delta messages."""
        llm = _make_mock_llm("RISK_LEVEL: low\nOVERALL_SCORE: 8.5\nVERDICT: approve")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "test", "messages": [SystemMessage(content="sys")]}
        result = await nodes["executive_verdict"](state)

        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], HumanMessage)
        assert isinstance(result["messages"][1], AIMessage)


class TestMessageAccumulation:
    """Tests for the messages accumulation pattern across nodes."""

    async def test_messages_grow_across_sequential_nodes(self, mock_config):
        """Messages should accumulate across sequential node calls, simulating the graph."""
        llm = _make_mock_llm("ARCHITECTURE_STYLE: serverless\nINDUSTRY: edtech\nCLOUD_PROVIDER: gcp")
        nodes = create_node_functions(llm, mock_config)

        # Simulate node 1: architecture_intake
        state_1 = {"query": "A serverless education platform on GCP"}
        result_1 = await nodes["architecture_intake"](state_1)
        all_messages = result_1["messages"]  # [SystemMessage, HumanMessage, AIMessage]
        assert len(all_messages) == 3

        # Simulate node 2: component_deep_dive (sees all previous messages)
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="COMPONENT_RISK: low"))
        state_2 = {"query": "test", "messages": all_messages}
        result_2 = await nodes["component_deep_dive"](state_2)

        # Delta should be 2 messages
        assert len(result_2["messages"]) == 2

        # Full accumulated should be 5 messages
        all_messages = all_messages + result_2["messages"]
        assert len(all_messages) == 5

        # Verify LLM was called with 4 messages (3 existing + 1 new)
        call_args = llm.ainvoke.call_args[0][0]
        assert len(call_args) == 4

    async def test_full_pipeline_message_count(self, mock_config):
        """Running all 7 nodes should produce 15 total messages (1 sys + 7 human + 7 ai)."""
        nodes_to_run = [
            "architecture_intake",
            "component_deep_dive",
            "security_posture",
            "reliability_scaling",
            "cost_efficiency",
            "compliance_gaps",
            "executive_verdict",
        ]

        response_texts = [
            "ARCHITECTURE_STYLE: microservices\nINDUSTRY: fintech\nCLOUD_PROVIDER: aws",
            "COMPONENT_RISK: medium",
            "SECURITY_RISK: high",
            "RELIABILITY_RISK: medium",
            "COST_RISK: low",
            "COMPLIANCE_RISK: high",
            "RISK_LEVEL: high\nOVERALL_SCORE: 5.5\nVERDICT: revise",
        ]

        all_messages = []
        state = {"query": "A microservices fintech platform on AWS", "messages": []}

        for i, node_name in enumerate(nodes_to_run):
            llm = _make_mock_llm(response_texts[i])
            nodes = create_node_functions(llm, mock_config)

            state["messages"] = all_messages
            result = await nodes[node_name](state)
            all_messages = all_messages + result["messages"]

        # 1 SystemMessage + 7 HumanMessages + 7 AIMessages = 15
        assert len(all_messages) == 15
        assert isinstance(all_messages[0], SystemMessage)

        # Verify alternating Human/AI pattern after system message
        for i in range(1, 15, 2):
            assert isinstance(all_messages[i], HumanMessage), f"Expected HumanMessage at index {i}"
            assert isinstance(all_messages[i + 1], AIMessage), f"Expected AIMessage at index {i + 1}"
