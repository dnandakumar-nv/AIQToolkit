# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utility helpers for ranking incident-response runbooks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-/]+")
_STOP_WORDS = {
    "the",
    "and",
    "a",
    "of",
    "for",
    "to",
    "in",
    "on",
    "is",
    "with",
    "an",
    "be",
    "by",
    "or",
    "that",
    "from",
    "as",
    "it",
    "this",
    "at",
    "any",
    "into",
    "your",
}


@dataclass(slots=True)
class Runbook:
    """In-memory representation of an incident response runbook."""

    identifier: str
    title: str
    summary: str
    key_steps: list[str]
    tags: list[str]
    severity: str
    detection_signals: list[str]

    @property
    def searchable_text(self) -> str:
        """Aggregate textual fields that should be considered during retrieval."""

        return "\n".join([
            self.title,
            self.summary,
            "\n".join(self.key_steps),
            " ".join(self.tags),
            " ".join(self.detection_signals),
            self.severity,
        ])


def _tokenize(text: str) -> list[str]:
    tokens = [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]
    return [token for token in tokens if token not in _STOP_WORDS]


def _to_counter(tokens: Iterable[str]) -> Counter[str]:
    counter: Counter[str] = Counter(tokens)
    total = sum(counter.values()) or 1
    for key in list(counter):
        counter[key] /= total
    return counter


def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
    common = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def score_runbook(query: str, runbook: Runbook) -> tuple[float, dict[str, list[str] | float | bool]]:
    """Score a runbook against a user query.

    Returns the similarity score along with an explanation payload that can be
    surfaced to the agent.
    """

    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(runbook.searchable_text)

    if not query_tokens or not doc_tokens:
        return 0.0, {"matched_terms": [], "tag_matches": [], "detection_matches": [], "severity_match": False}

    similarity = _cosine_similarity(_to_counter(query_tokens), _to_counter(doc_tokens))

    tag_matches = [tag for tag in runbook.tags if tag.lower() in query_tokens]
    detection_matches = [signal for signal in runbook.detection_signals if signal.lower() in query_tokens]
    severity_match = runbook.severity.lower() in query_tokens

    # Provide interpretable boost factors so we can justify the ranking to the agent.
    boost = 0.0
    if tag_matches:
        boost += 0.1 * len(tag_matches)
    if detection_matches:
        boost += 0.05 * len(detection_matches)
    if severity_match:
        boost += 0.15

    final_score = similarity + boost

    return final_score, {
        "similarity": round(similarity, 3),
        "matched_terms": sorted(set(query_tokens).intersection(doc_tokens)),
        "tag_matches": tag_matches,
        "detection_matches": detection_matches,
        "severity_match": severity_match,
        "boost_applied": round(boost, 3),
    }


def format_ranked_runbook(runbook: Runbook, score: float, explanation: dict[str, list[str] | float | bool]) -> dict[str, object]:
    """Structure runbook details so that downstream prompts can reason over them."""

    return {
        "id": runbook.identifier,
        "title": runbook.title,
        "summary": runbook.summary,
        "key_steps": runbook.key_steps,
        "tags": runbook.tags,
        "severity": runbook.severity,
        "detection_signals": runbook.detection_signals,
        "score": round(score, 3),
        "evidence": explanation,
    }

