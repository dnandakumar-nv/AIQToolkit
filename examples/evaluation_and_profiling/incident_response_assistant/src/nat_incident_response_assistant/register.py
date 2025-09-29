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

"""Registers the incident response assistant tooling for NAT."""

from __future__ import annotations

import json
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .scoring import Runbook
from .scoring import format_ranked_runbook
from .scoring import score_runbook

logger = logging.getLogger(__name__)


class IncidentRunbookSearchConfig(FunctionBaseConfig, name="incident_runbook_search"):
    """Configuration for the local runbook search tool."""

    runbook_path: Path
    top_k: int = 3


@lru_cache(maxsize=4)
def _load_runbooks(runbook_path: Path) -> tuple[Runbook, ...]:
    """Load runbook definitions from disk and convert to dataclasses."""

    with runbook_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    runbooks: list[Runbook] = []
    for entry in data:
        runbooks.append(
            Runbook(
                identifier=entry["id"],
                title=entry["title"],
                summary=entry["summary"],
                key_steps=list(entry["key_steps"]),
                tags=list(entry.get("tags", [])),
                severity=entry.get("severity", "unknown"),
                detection_signals=list(entry.get("detection_signals", [])),
            )
        )
    logger.debug("Loaded %s runbooks from %s", len(runbooks), runbook_path)
    return tuple(runbooks)


@register_function(config_type=IncidentRunbookSearchConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def incident_runbook_search(config: IncidentRunbookSearchConfig, builder: Builder) -> Any:
    """Expose the runbook search capability as a tool."""

    # The builder argument is not used in this simple example but is kept for
    # consistency with other functions and future extensibility.
    _ = builder

    runbooks = _load_runbooks(config.runbook_path)

    async def _search_incident_playbooks(query: str) -> str:
        if not query.strip():
            return json.dumps({"error": "Empty query provided"})

        scored = [
            (runbook, *score_runbook(query, runbook))
            for runbook in runbooks
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        top_results = [
            format_ranked_runbook(runbook=item[0], score=item[1], explanation=item[2])
            for item in scored[: config.top_k]
        ]

        payload = {
            "query": query,
            "results": top_results,
            "runbooks_considered": len(runbooks),
        }
        return json.dumps(payload, indent=2)

    description = (
        "Searches the local incident response runbooks and returns the most "
        "relevant playbooks with evidence for the match."
    )

    yield FunctionInfo.from_fn(
        fn=_search_incident_playbooks,
        description=description,
    )


