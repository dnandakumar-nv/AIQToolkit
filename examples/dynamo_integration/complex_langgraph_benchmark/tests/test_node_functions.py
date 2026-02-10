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

Verifies that each node function produces correct state updates
when given mock LLM responses.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from complex_langgraph_benchmark.graph_nodes import create_node_functions


@pytest.fixture(name="mock_config")
def fixture_mock_config():
    """Create a mock config with default values."""
    return SimpleNamespace(
        max_quality_retries=2,
        max_synthesis_retries=2,
        quality_threshold=0.7,
        decision_only=True,
    )


def _make_mock_llm(response_text: str):
    """Create a mock LLM returning a specific response."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=response_text))
    return llm


class TestClassifyIncident:
    """Tests for the classify_incident node."""

    async def test_parses_severity_and_category(self, mock_config):
        """classify_incident should parse severity, category, and reasoning from LLM response."""
        llm = _make_mock_llm("SEVERITY: critical\nCATEGORY: security\nREASONING: Database breach detected")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "Unauthorized access to production database detected"}
        result = await nodes["classify_incident"](state)

        assert result["severity"] == "critical"
        assert result["category"] == "security"
        assert "Database breach detected" in result["classification_reasoning"]
        assert result["quality_attempts"] == 0
        assert result["synthesis_attempts"] == 0

    async def test_defaults_on_unparseable_response(self, mock_config):
        """classify_incident should use defaults when LLM response is unstructured."""
        llm = _make_mock_llm("This is a generic response without structured fields.")
        nodes = create_node_functions(llm, mock_config)

        state = {"query": "Something happened"}
        result = await nodes["classify_incident"](state)

        assert result["severity"] == "standard"
        assert result["category"] == "mixed"


class TestAnalyzeBranch:
    """Tests for the analyze_branch node."""

    async def test_security_branch(self, mock_config):
        """analyze_branch should use security prompt for security branch."""
        llm = _make_mock_llm("Security analysis: potential SQL injection vector identified.")
        nodes = create_node_functions(llm, mock_config)

        state = {
            "branch": "security",
            "query": "Unusual database queries detected",
            "severity": "critical",
            "category": "security",
            "classification_reasoning": "Suspicious patterns",
        }
        result = await nodes["analyze_branch"](state)

        assert len(result["analysis_results"]) == 1
        assert result["analysis_results"][0]["branch"] == "security"
        assert "SQL injection" in result["analysis_results"][0]["findings"]

    async def test_performance_branch(self, mock_config):
        """analyze_branch should use performance prompt for performance branch."""
        llm = _make_mock_llm("Performance analysis: high CPU utilization at 95%.")
        nodes = create_node_functions(llm, mock_config)

        state = {
            "branch": "performance",
            "query": "API response times degraded",
            "severity": "standard",
            "category": "performance",
            "classification_reasoning": "Slow responses",
        }
        result = await nodes["analyze_branch"](state)

        assert result["analysis_results"][0]["branch"] == "performance"

    async def test_infrastructure_branch(self, mock_config):
        """analyze_branch should use infrastructure prompt for infrastructure branch."""
        llm = _make_mock_llm("Infrastructure analysis: disk utilization at 98%.")
        nodes = create_node_functions(llm, mock_config)

        state = {
            "branch": "infrastructure",
            "query": "Storage alerts firing",
            "severity": "standard",
            "category": "infrastructure",
            "classification_reasoning": "Disk space",
        }
        result = await nodes["analyze_branch"](state)

        assert result["analysis_results"][0]["branch"] == "infrastructure"


class TestAggregateFinddings:
    """Tests for the aggregate_findings node."""

    async def test_merges_multiple_results(self, mock_config):
        """aggregate_findings should merge all analysis results."""
        llm = _make_mock_llm("")
        nodes = create_node_functions(llm, mock_config)

        state = {
            "analysis_results": [
                {"branch": "security", "findings": "No threats found."},
                {"branch": "performance", "findings": "CPU at 50%."},
                {"branch": "infrastructure", "findings": "All systems nominal."},
            ],
        }
        result = await nodes["aggregate_findings"](state)

        assert "SECURITY ANALYSIS" in result["aggregated_findings"]
        assert "PERFORMANCE ANALYSIS" in result["aggregated_findings"]
        assert "INFRASTRUCTURE ANALYSIS" in result["aggregated_findings"]


class TestAssessQuality:
    """Tests for the assess_quality node."""

    async def test_parses_score_and_feedback(self, mock_config):
        """assess_quality should parse score and feedback from LLM response."""
        llm = _make_mock_llm("SCORE: 0.85\nFEEDBACK: Analysis is thorough and actionable.")
        nodes = create_node_functions(llm, mock_config)

        state = {
            "query": "Test incident",
            "aggregated_findings": "Test findings",
            "quality_feedback": "None",
            "quality_attempts": 0,
        }
        result = await nodes["assess_quality"](state)

        assert result["quality_score"] == 0.85
        assert "thorough" in result["quality_feedback"]
        assert result["quality_attempts"] == 1

    async def test_clamps_score_to_valid_range(self, mock_config):
        """assess_quality should clamp score between 0.0 and 1.0."""
        llm = _make_mock_llm("SCORE: 1.5\nFEEDBACK: Overconfident score.")
        nodes = create_node_functions(llm, mock_config)

        state = {
            "query": "Test",
            "aggregated_findings": "Test",
            "quality_feedback": "None",
            "quality_attempts": 0,
        }
        result = await nodes["assess_quality"](state)

        assert result["quality_score"] == 1.0


class TestExtractActions:
    """Tests for the extract_actions node."""

    async def test_parses_action_blocks(self, mock_config):
        """extract_actions should parse ACTION/PARAMS/REASON blocks."""
        llm = _make_mock_llm(
            "ACTION: create_ticket\n"
            "PARAMS: title=DB Outage, priority=P0, assignee_team=dba-team\n"
            "REASON: Track the incident\n"
            "ACTION: escalate_incident\n"
            "PARAMS: level=L2, reason=DB down, urgency=high\n"
            "REASON: Needs immediate attention")
        nodes = create_node_functions(llm, mock_config)

        state = {
            "query": "Database is down",
            "incident_id": "INC-001",
            "severity": "critical",
            "category": "infrastructure",
            "draft_report": "Database failure report",
        }
        result = await nodes["extract_actions"](state)

        assert len(result["tool_calls_made"]) == 2
        assert result["tool_calls_made"][0]["tool"] == "create_ticket"
        assert result["tool_calls_made"][0]["parameters"]["priority"] == "P0"
        assert result["tool_calls_made"][1]["tool"] == "escalate_incident"


class TestRiskAssessment:
    """Tests for the risk_assessment node."""

    async def test_parses_risk_level(self, mock_config):
        """risk_assessment should parse risk level and reasoning."""
        llm = _make_mock_llm("RISK_LEVEL: high\nREASONING: Potential for data loss if not addressed within 2 hours.")
        nodes = create_node_functions(llm, mock_config)

        state = {
            "query": "Test incident",
            "incident_id": "INC-001",
            "severity": "critical",
            "category": "infrastructure",
            "draft_report": "Test report",
            "recommended_actions": ["create_ticket: Track the issue"],
        }
        result = await nodes["risk_assessment"](state)

        assert result["risk_level"] == "high"
        assert "data loss" in result["risk_reasoning"]


class TestFinalSummary:
    """Tests for the final_summary node."""

    async def test_generates_response(self, mock_config):
        """final_summary should generate a final response string."""
        llm = _make_mock_llm("Incident INC-001 has been fully analyzed and resolved.")
        nodes = create_node_functions(llm, mock_config)

        state = {
            "incident_id": "INC-001",
            "severity": "critical",
            "category": "infrastructure",
            "risk_level": "high",
            "draft_report": "Full report here",
            "risk_reasoning": "Risk is high due to data exposure",
            "recommended_actions": ["create_ticket: Track"],
        }
        result = await nodes["final_summary"](state)

        assert "INC-001" in result["final_response"]
        assert len(result["final_response"]) > 0
