from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

IntentLabel = Literal[
    "hypothesis_retrieval",
    "subproblem_retrieval",
    "constraint_focus",
    "implementation_debug",
    "concept_explanation",
]


@dataclass(slots=True)
class QueryArtifacts:
    raw_query: str
    image_summary: str = ""
    merged_query: str = ""
    intent: IntentLabel = "concept_explanation"
    rewritten_query: str = ""
    rag_hits: list[dict] = field(default_factory=list)
    rag_context: str = ""
    answer: str = ""
