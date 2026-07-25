"""Project-scoped Jira identity resolution for presentation payloads.

Jira account IDs are stable machine identifiers; they must never be used as a
display label when a synced team roster can resolve them.
"""
from __future__ import annotations

import re

from app.models.models import ProjectTeamMember


_OPAQUE_ID = re.compile(r"^[a-f0-9]{16,}$|^[A-Za-z0-9_-]{12,}$")


class IdentityResolver:
    def __init__(self, members: list[ProjectTeamMember]):
        self._names = {
            str(member.jira_account_id): member.display_name
            for member in members
            if member.display_name and member.display_name.strip()
        }

    @classmethod
    def for_project(cls, db, project_id: int) -> "IdentityResolver":
        query = db.query(ProjectTeamMember).filter_by(project_id=project_id)
        # Lightweight service tests may provide only count-capable query
        # doubles; absence of a roster simply means live display names remain.
        return cls(query.all() if hasattr(query, "all") else [])

    def resolve(self, value: object) -> str:
        if value is None or not str(value).strip() or str(value).casefold() == "unassigned":
            return "Unassigned"
        raw = str(value).strip()
        # A Jira display name may already be in the live issue snapshot.
        if raw in self._names.values():
            return raw
        resolved = self._names.get(raw)
        if resolved:
            return resolved
        # Never leak opaque account/database identifiers into presentation.
        return "Unassigned" if _OPAQUE_ID.fullmatch(raw) else raw

    def public_map(self) -> dict[str, str]:
        return dict(self._names)
