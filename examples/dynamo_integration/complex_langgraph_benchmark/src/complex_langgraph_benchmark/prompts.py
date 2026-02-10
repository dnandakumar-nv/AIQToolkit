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
Prompt templates for the Complex LangGraph Benchmark Agent.

Each constant is a string template with {placeholder} variables that get
formatted with IncidentState fields at runtime.
"""

CLASSIFY_INCIDENT = """You are an IT incident response classifier. Analyze the following incident report and classify it.

Incident Report:
{query}

Classify this incident by providing:
1. **Severity**: One of "critical", "standard", or "low_priority"
   - critical: Service outage, data breach, security compromise, complete system failure
   - standard: Performance degradation, partial outage, non-critical errors, elevated error rates
   - low_priority: Informational alerts, minor issues, routine maintenance needs, cosmetic problems

2. **Category**: One of "security", "performance", "infrastructure", or "mixed"
   - security: Authentication failures, unauthorized access, data breaches, vulnerability exploits
   - performance: Slow response times, high latency, resource exhaustion, throughput degradation
   - infrastructure: Hardware failures, network issues, DNS problems, storage failures, deployment issues
   - mixed: Incidents spanning multiple categories or cascading failures

3. **Reasoning**: A brief explanation of your classification decision.

Respond in exactly this format:
SEVERITY: <severity>
CATEGORY: <category>
REASONING: <reasoning>"""

SECURITY_ANALYSIS = """You are a cybersecurity incident analyst. Analyze the following incident from a security perspective.

Incident Report:
{query}

Severity: {severity}
Category: {category}
Classification Reasoning: {classification_reasoning}

Perform a thorough security analysis:
1. Identify potential attack vectors or security implications
2. Assess data exposure risk
3. Check for indicators of compromise (IoCs)
4. Evaluate access control implications
5. Recommend immediate security containment actions

Provide your analysis in a structured format with clear findings and recommendations."""

PERFORMANCE_ANALYSIS = """You are a systems performance engineer. Analyze the following incident from a performance perspective.

Incident Report:
{query}

Severity: {severity}
Category: {category}
Classification Reasoning: {classification_reasoning}

Perform a thorough performance analysis:
1. Identify performance bottlenecks and degradation patterns
2. Assess impact on system throughput and latency
3. Evaluate resource utilization (CPU, memory, I/O, network)
4. Determine blast radius and affected services
5. Recommend performance optimization and remediation steps

Provide your analysis in a structured format with clear findings and recommendations."""

INFRASTRUCTURE_ANALYSIS = """You are an infrastructure reliability engineer. Analyze the following incident from an infrastructure perspective.

Incident Report:
{query}

Severity: {severity}
Category: {category}
Classification Reasoning: {classification_reasoning}

Perform a thorough infrastructure analysis:
1. Identify affected infrastructure components (servers, networks, storage, DNS)
2. Assess redundancy and failover status
3. Evaluate deployment and configuration factors
4. Determine cascading failure risks
5. Recommend infrastructure remediation and hardening steps

Provide your analysis in a structured format with clear findings and recommendations."""

ASSESS_QUALITY = """You are a quality assurance reviewer for incident analysis reports. Evaluate the quality of the following aggregated findings.

Original Incident:
{query}

Aggregated Findings:
{aggregated_findings}

Previous Quality Feedback (if any):
{quality_feedback}

Quality Attempt: {quality_attempts}

Score the analysis quality from 0.0 to 1.0 based on:
1. Completeness: Are all relevant aspects of the incident addressed?
2. Specificity: Are findings specific rather than generic?
3. Actionability: Are the recommendations concrete and actionable?
4. Consistency: Do the findings align with the incident severity and category?
5. Depth: Is the analysis sufficiently thorough for the incident type?

Respond in exactly this format:
SCORE: <float between 0.0 and 1.0>
FEEDBACK: <detailed feedback explaining the score and areas for improvement>"""

SYNTHESIZE_REPORT = """You are an incident response report writer. Synthesize a comprehensive incident report from the analysis findings.

Original Incident:
{query}

Incident ID: {incident_id}
Severity: {severity}
Category: {category}

Aggregated Analysis Findings:
{aggregated_findings}

Previous Critique Feedback (if any):
{critique_feedback}

Synthesis Attempt: {synthesis_attempts}

Write a comprehensive incident report that includes:
1. Executive Summary
2. Incident Timeline and Impact
3. Root Cause Analysis
4. Affected Systems and Services
5. Remediation Steps (Immediate and Long-term)
6. Lessons Learned

The report should be clear, actionable, and appropriate for both technical and management audiences."""

CRITIQUE_REPORT = """You are a senior incident response manager reviewing a draft incident report for quality and completeness.

Original Incident:
{query}

Severity: {severity}
Category: {category}

Draft Report:
{draft_report}

Review the report critically and determine if it meets the following criteria:
1. Is the executive summary clear and accurate?
2. Does the root cause analysis identify the actual root cause?
3. Are remediation steps specific and prioritized?
4. Is the report appropriate for stakeholder communication?
5. Are there any factual inconsistencies or gaps?

If the report needs revision, respond with:
VERDICT: REVISE
FEEDBACK: <specific feedback for improvement>

If the report is acceptable, respond with:
VERDICT: ACCEPT
FEEDBACK: <brief confirmation of quality>"""

EXTRACT_ACTIONS = """You are an incident response coordinator. Based on the incident analysis and report, determine the specific actions that need to be taken.

Original Incident:
{query}

Incident ID: {incident_id}
Severity: {severity}
Category: {category}

Incident Report:
{draft_report}

Based on the analysis, determine which of the following actions should be taken. For each action, provide the specific parameters.

Available Actions:
1. create_ticket - Create a support/incident ticket (title, priority P0-P3, assignee_team)
2. escalate_incident - Escalate to on-call team (level L1-L3, reason, urgency)
3. notify_team - Send notification (team, channel: slack/email/pager, message)
4. schedule_followup - Schedule follow-up action (type: review/postmortem/recheck, delay_hours)
5. check_system_status - Check system health (system_name, check_type)
6. apply_mitigation - Apply mitigation action (action, target_system, rollback_plan)

List each action you recommend in this format:
ACTION: <tool_name>
PARAMS: <param1>=<value1>, <param2>=<value2>, ...
REASON: <why this action is needed>

List all applicable actions."""

RISK_ASSESSMENT = """You are a risk assessment specialist. Evaluate the overall risk level for this incident based on the full analysis.

Original Incident:
{query}

Incident ID: {incident_id}
Severity: {severity}
Category: {category}
Incident Report:
{draft_report}

Actions Taken:
{recommended_actions}

Assess the current risk level considering:
1. Likelihood of recurrence
2. Potential for escalation or cascading failures
3. Data integrity and confidentiality impact
4. Business continuity impact
5. Effectiveness of applied mitigations

Respond in exactly this format:
RISK_LEVEL: <one of: critical, high, medium, low>
REASONING: <detailed risk assessment reasoning>"""

FINAL_SUMMARY = """You are an incident response communicator. Create a final comprehensive summary for this incident.

Incident ID: {incident_id}
Severity: {severity}
Category: {category}
Risk Level: {risk_level}

Incident Report:
{draft_report}

Risk Assessment:
{risk_reasoning}

Actions Recommended:
{recommended_actions}

Create a clear, concise final summary that:
1. States the incident status and current risk level
2. Summarizes key findings from the analysis
3. Lists completed and pending actions
4. Provides next steps and timeline
5. Includes any escalation or communication needs

This summary should be suitable as the final response to the incident reporter."""
