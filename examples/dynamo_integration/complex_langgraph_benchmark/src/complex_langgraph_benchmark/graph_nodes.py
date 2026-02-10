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
Graph node functions for the Complex LangGraph Benchmark Agent.

Contains all 12 async node functions and 3 conditional edge functions
for the IT Incident Response workflow graph.
"""

import logging
import re
import uuid

from langgraph.types import Send

from nat.profiler.decorators.latency import LatencySensitivity
from nat.profiler.decorators.latency import latency_sensitive

from . import prompts
from .graph_state import IncidentState

logger = logging.getLogger(__name__)


def create_node_functions(llm, config):
    """
    Create all graph node functions with the LLM instance captured via closure.

    Args:
        llm: LangChain-wrapped LLM instance from the NAT builder.
        config: Agent configuration with thresholds and retry limits.

    Returns:
        Dictionary mapping node names to async node functions, plus conditional edge functions.
    """

    # =========================================================================
    # Node 1: classify_incident — LLM call to classify severity + category
    # =========================================================================
    @latency_sensitive(LatencySensitivity.HIGH)
    async def classify_incident(state: IncidentState) -> dict:
        """Classify the incident by severity and category using LLM."""
        query = state["query"]
        incident_id = state.get("incident_id", str(uuid.uuid4())[:8])

        prompt_text = prompts.CLASSIFY_INCIDENT.format(query=query)
        response = await llm.ainvoke(prompt_text)
        content = response.content if hasattr(response, 'content') else str(response)

        # Parse the structured response
        severity = "standard"
        category = "mixed"
        reasoning = content

        severity_match = re.search(r'SEVERITY:\s*(\w+)', content, re.IGNORECASE)
        if severity_match:
            severity = severity_match.group(1).lower()
            if severity not in ("critical", "standard", "low_priority"):
                severity = "standard"

        category_match = re.search(r'CATEGORY:\s*(\w+)', content, re.IGNORECASE)
        if category_match:
            category = category_match.group(1).lower()
            if category not in ("security", "performance", "infrastructure", "mixed"):
                category = "mixed"

        reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        logger.info("Classified incident %s: severity=%s, category=%s", incident_id, severity, category)

        return {
            "incident_id": incident_id,
            "severity": severity,
            "category": category,
            "classification_reasoning": reasoning,
            "quality_attempts": 0,
            "synthesis_attempts": 0,
            "analysis_results": [],
            "tool_calls_made": [],
            "recommended_actions": [],
            "messages": [],
        }

    # =========================================================================
    # Conditional edge: fan_out_router — Returns Send() to dispatch 3 parallel branches
    # Used as a conditional edge function (not a node) because it returns Send objects
    # =========================================================================
    @latency_sensitive(LatencySensitivity.LOW)
    def fan_out_router(state: IncidentState) -> list[Send]:
        """Dispatch parallel analysis branches via Send()."""
        logger.info("Dispatching 3 parallel analysis branches for incident %s", state.get("incident_id", "unknown"))

        base_context = {
            "query": state["query"],
            "severity": state["severity"],
            "category": state["category"],
            "classification_reasoning": state["classification_reasoning"],
        }

        return [
            Send("analyze_branch", {**base_context, "branch": "security"}),
            Send("analyze_branch", {**base_context, "branch": "performance"}),
            Send("analyze_branch", {**base_context, "branch": "infrastructure"}),
        ]

    # =========================================================================
    # Node 3/4/5: analyze_branch — Single node handling all 3 analysis branches
    # =========================================================================
    @latency_sensitive(LatencySensitivity.MEDIUM)
    async def analyze_branch(state: dict) -> dict:
        """Execute analysis for a specific branch (security/performance/infrastructure)."""
        branch = state["branch"]
        query = state["query"]
        severity = state["severity"]
        category = state["category"]
        classification_reasoning = state["classification_reasoning"]

        # Select prompt based on branch
        if branch == "security":
            prompt_text = prompts.SECURITY_ANALYSIS.format(
                query=query, severity=severity, category=category,
                classification_reasoning=classification_reasoning)
        elif branch == "performance":
            prompt_text = prompts.PERFORMANCE_ANALYSIS.format(
                query=query, severity=severity, category=category,
                classification_reasoning=classification_reasoning)
        else:  # infrastructure
            prompt_text = prompts.INFRASTRUCTURE_ANALYSIS.format(
                query=query, severity=severity, category=category,
                classification_reasoning=classification_reasoning)

        response = await llm.ainvoke(prompt_text)
        content = response.content if hasattr(response, 'content') else str(response)

        logger.info("Completed %s analysis (%d chars)", branch, len(content))

        return {
            "analysis_results": [{"branch": branch, "findings": content}],
        }

    # =========================================================================
    # Node 6: aggregate_findings — Pure logic: merge parallel results
    # =========================================================================
    @latency_sensitive(LatencySensitivity.LOW)
    async def aggregate_findings(state: IncidentState) -> dict:
        """Merge parallel analysis results into a single aggregated summary."""
        results = state.get("analysis_results", [])

        sections = []
        for result in results:
            branch = result.get("branch", "unknown")
            findings = result.get("findings", "No findings.")
            sections.append(f"=== {branch.upper()} ANALYSIS ===\n{findings}")

        aggregated = "\n\n".join(sections)
        logger.info("Aggregated %d analysis results (%d chars)", len(results), len(aggregated))

        return {"aggregated_findings": aggregated}

    # =========================================================================
    # Node 7: assess_quality — LLM call: score quality 0-1
    # =========================================================================
    @latency_sensitive(LatencySensitivity.MEDIUM)
    async def assess_quality(state: IncidentState) -> dict:
        """Assess the quality of aggregated findings using LLM."""
        quality_attempts = state.get("quality_attempts", 0)

        prompt_text = prompts.ASSESS_QUALITY.format(
            query=state["query"],
            aggregated_findings=state["aggregated_findings"],
            quality_feedback=state.get("quality_feedback", "None"),
            quality_attempts=quality_attempts,
        )

        response = await llm.ainvoke(prompt_text)
        content = response.content if hasattr(response, 'content') else str(response)

        # Parse score and feedback
        score = 0.8  # default
        feedback = content

        score_match = re.search(r'SCORE:\s*([\d.]+)', content)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                score = 0.8

        feedback_match = re.search(r'FEEDBACK:\s*(.+)', content, re.DOTALL)
        if feedback_match:
            feedback = feedback_match.group(1).strip()

        logger.info("Quality assessment: score=%.2f, attempt=%d", score, quality_attempts + 1)

        return {
            "quality_score": score,
            "quality_feedback": feedback,
            "quality_attempts": quality_attempts + 1,
        }

    # =========================================================================
    # Node 8: synthesize_report — LLM call: draft report
    # =========================================================================
    @latency_sensitive(LatencySensitivity.MEDIUM)
    async def synthesize_report(state: IncidentState) -> dict:
        """Synthesize a comprehensive incident report."""
        synthesis_attempts = state.get("synthesis_attempts", 0)

        prompt_text = prompts.SYNTHESIZE_REPORT.format(
            query=state["query"],
            incident_id=state["incident_id"],
            severity=state["severity"],
            category=state["category"],
            aggregated_findings=state["aggregated_findings"],
            critique_feedback=state.get("critique_feedback", "None"),
            synthesis_attempts=synthesis_attempts,
        )

        response = await llm.ainvoke(prompt_text)
        content = response.content if hasattr(response, 'content') else str(response)

        logger.info("Synthesized report (%d chars), attempt=%d", len(content), synthesis_attempts + 1)

        return {
            "draft_report": content,
            "synthesis_attempts": synthesis_attempts + 1,
        }

    # =========================================================================
    # Node 9: critique_report — LLM call: review draft
    # =========================================================================
    @latency_sensitive(LatencySensitivity.MEDIUM)
    async def critique_report(state: IncidentState) -> dict:
        """Critique the draft report and decide whether to accept or revise."""
        prompt_text = prompts.CRITIQUE_REPORT.format(
            query=state["query"],
            severity=state["severity"],
            category=state["category"],
            draft_report=state["draft_report"],
        )

        response = await llm.ainvoke(prompt_text)
        content = response.content if hasattr(response, 'content') else str(response)

        logger.info("Critique completed: %s", content[:100])

        return {"critique_feedback": content}

    # =========================================================================
    # Node 10: extract_actions — LLM call + stub tool calls
    # =========================================================================
    @latency_sensitive(LatencySensitivity.MEDIUM)
    async def extract_actions(state: IncidentState) -> dict:
        """Extract recommended actions and invoke stub tools."""
        prompt_text = prompts.EXTRACT_ACTIONS.format(
            query=state["query"],
            incident_id=state["incident_id"],
            severity=state["severity"],
            category=state["category"],
            draft_report=state["draft_report"],
        )

        response = await llm.ainvoke(prompt_text)
        content = response.content if hasattr(response, 'content') else str(response)

        # Parse actions from the response
        actions = []
        tool_calls = []

        # Find all ACTION blocks
        action_blocks = re.findall(
            r'ACTION:\s*(\w+)\s*\nPARAMS:\s*(.+?)(?:\nREASON:\s*(.+?))?(?=\nACTION:|\Z)',
            content,
            re.DOTALL,
        )

        for tool_name, params_str, reason in action_blocks:
            tool_name = tool_name.strip()
            reason = reason.strip() if reason else ""

            # Parse key=value params
            params = {}
            for param_pair in params_str.strip().split(","):
                param_pair = param_pair.strip()
                if "=" in param_pair:
                    key, value = param_pair.split("=", 1)
                    params[key.strip()] = value.strip()

            tool_calls.append({"tool": tool_name, "parameters": params})
            actions.append(f"{tool_name}: {reason}")

        # Fallback if no structured actions found
        if not actions:
            actions = [content[:200]]

        logger.info("Extracted %d actions with %d tool calls", len(actions), len(tool_calls))

        return {
            "tool_calls_made": tool_calls,
            "recommended_actions": actions,
        }

    # =========================================================================
    # Node 11: risk_assessment — LLM call: final risk evaluation
    # =========================================================================
    @latency_sensitive(LatencySensitivity.MEDIUM)
    async def risk_assessment(state: IncidentState) -> dict:
        """Perform final risk assessment."""
        prompt_text = prompts.RISK_ASSESSMENT.format(
            query=state["query"],
            incident_id=state["incident_id"],
            severity=state["severity"],
            category=state["category"],
            draft_report=state["draft_report"],
            recommended_actions="\n".join(state.get("recommended_actions", [])),
        )

        response = await llm.ainvoke(prompt_text)
        content = response.content if hasattr(response, 'content') else str(response)

        # Parse risk level and reasoning
        risk_level = "medium"
        risk_reasoning = content

        level_match = re.search(r'RISK_LEVEL:\s*(\w+)', content, re.IGNORECASE)
        if level_match:
            risk_level = level_match.group(1).lower()
            if risk_level not in ("critical", "high", "medium", "low"):
                risk_level = "medium"

        reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            risk_reasoning = reasoning_match.group(1).strip()

        logger.info("Risk assessment: level=%s", risk_level)

        return {
            "risk_level": risk_level,
            "risk_reasoning": risk_reasoning,
        }

    # =========================================================================
    # Node 12: final_summary — LLM call: comprehensive response
    # =========================================================================
    @latency_sensitive(LatencySensitivity.HIGH)
    async def final_summary(state: IncidentState) -> dict:
        """Generate the final comprehensive summary response."""
        prompt_text = prompts.FINAL_SUMMARY.format(
            incident_id=state["incident_id"],
            severity=state["severity"],
            category=state["category"],
            risk_level=state["risk_level"],
            draft_report=state["draft_report"],
            risk_reasoning=state["risk_reasoning"],
            recommended_actions="\n".join(state.get("recommended_actions", [])),
        )

        response = await llm.ainvoke(prompt_text)
        content = response.content if hasattr(response, 'content') else str(response)

        logger.info("Final summary generated (%d chars)", len(content))

        return {"final_response": content}

    # =========================================================================
    # Conditional edge functions
    # =========================================================================
    @latency_sensitive(LatencySensitivity.LOW)
    def quality_gate(state: IncidentState) -> list[Send] | str:
        """Check quality score — retry analysis via Send() if below threshold, else proceed."""
        score = state.get("quality_score", 1.0)
        attempts = state.get("quality_attempts", 0)
        max_retries = getattr(config, 'max_quality_retries', 2)
        threshold = getattr(config, 'quality_threshold', 0.7)

        if score < threshold and attempts < max_retries:
            logger.info("Quality gate: score=%.2f < %.2f, retrying (attempt %d/%d)",
                        score, threshold, attempts, max_retries)
            # Re-dispatch parallel analysis via Send()
            base_context = {
                "query": state["query"],
                "severity": state["severity"],
                "category": state["category"],
                "classification_reasoning": state["classification_reasoning"],
            }
            return [
                Send("analyze_branch", {**base_context, "branch": "security"}),
                Send("analyze_branch", {**base_context, "branch": "performance"}),
                Send("analyze_branch", {**base_context, "branch": "infrastructure"}),
            ]

        logger.info("Quality gate: score=%.2f, proceeding to synthesis", score)
        return "synthesize_report"

    @latency_sensitive(LatencySensitivity.LOW)
    def critique_gate(state: IncidentState) -> str:
        """Check critique verdict — retry synthesis if revision needed and under retry limit."""
        feedback = state.get("critique_feedback", "")
        attempts = state.get("synthesis_attempts", 0)
        max_retries = getattr(config, 'max_synthesis_retries', 2)

        if "REVISE" in feedback.upper() and attempts < max_retries:
            logger.info("Critique gate: revision needed, retrying (attempt %d/%d)", attempts, max_retries)
            return "synthesize_report"

        logger.info("Critique gate: accepted, proceeding to actions")
        return "extract_actions"

    return {
        "classify_incident": classify_incident,
        "fan_out_router": fan_out_router,  # conditional edge function, not a node
        "analyze_branch": analyze_branch,
        "aggregate_findings": aggregate_findings,
        "assess_quality": assess_quality,
        "synthesize_report": synthesize_report,
        "critique_report": critique_report,
        "extract_actions": extract_actions,
        "risk_assessment": risk_assessment,
        "final_summary": final_summary,
        "quality_gate": quality_gate,
        "critique_gate": critique_gate,
    }
