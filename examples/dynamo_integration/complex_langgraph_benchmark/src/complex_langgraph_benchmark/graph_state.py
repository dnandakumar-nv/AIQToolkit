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
State schema for the Complex LangGraph Benchmark Agent.

Defines the IncidentState TypedDict used by all graph nodes to communicate
through the LangGraph state machine.
"""

import operator
from typing import Annotated
from typing import TypedDict


class IncidentState(TypedDict):
    """
    State for the IT Incident Response agent graph.

    Fields:
        query: The original incident description from the user.
        incident_id: Unique identifier for this incident.
        severity: Classification result - "critical", "standard", or "low_priority".
        category: Classification result - "security", "performance", "infrastructure", or "mixed".
        classification_reasoning: LLM reasoning for the classification decision.
        analysis_results: Accumulated results from parallel analysis branches (fan-in via operator.add).
        aggregated_findings: Merged summary of all parallel analysis results.
        quality_score: Quality assessment score (0.0 to 1.0) from the quality gate.
        quality_feedback: Feedback from quality assessment explaining the score.
        quality_attempts: Number of quality gate retry attempts (max 2).
        draft_report: The synthesized incident report draft.
        critique_feedback: Feedback from the critique step on the draft report.
        synthesis_attempts: Number of synthesis retry attempts (max 2).
        tool_calls_made: List of tool call intents captured during extract_actions.
        recommended_actions: List of recommended remediation actions.
        risk_level: Final risk level assessment - "critical", "high", "medium", or "low".
        risk_reasoning: Reasoning behind the risk level assessment.
        final_response: The comprehensive final response to return.
        messages: Message list for NAT framework compatibility.
    """

    query: str
    incident_id: str
    severity: str
    category: str
    classification_reasoning: str
    analysis_results: Annotated[list[dict], operator.add]
    aggregated_findings: str
    quality_score: float
    quality_feedback: str
    quality_attempts: int
    draft_report: str
    critique_feedback: str
    synthesis_attempts: int
    tool_calls_made: list[dict]
    recommended_actions: list[str]
    risk_level: str
    risk_reasoning: str
    final_response: str
    messages: list
