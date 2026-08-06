"""Agent state passed between LangGraph nodes."""
from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict


class ChatState(TypedDict, total=False):
    # Inputs
    question: str
    original_question: str
    role: str
    active_project_id: int
    allowed_project_ids: List[int]
    doc_types: Optional[List[str]]

    # Working state
    authorized: bool
    in_scope: bool
    passages: List[dict]
    best_score: float
    attempts: int
    grounded: bool

    # Outputs
    answer: str
    citations: List[dict]
    status: str                       # answered | not_available | denied | out_of_scope

    # Audit — accumulated across nodes
    trace: Annotated[List[dict], operator.add]
