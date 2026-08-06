"""FR-13 API — Agentic Document Assistant.

Endpoints:
  GET  /knowledge/projects                        list authorized projects (selector, FR-17)
  POST /knowledge/projects/{id}/sync              sync Confluence/local docs into the KB
  POST /knowledge/projects/{id}/chat              ask the agentic chatbot
  GET  /knowledge/conversations/{id}/messages     conversation history
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.utils.database import get_db
from app.api.auth import get_current_user
from app.models.models import User
from app.knowledge.models import KbConversation, KbMessage
from app.knowledge.service import KnowledgeAssistant, AuthorizationError

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[int] = None
    doc_types: Optional[List[str]] = None


class ConfluenceConnectionRequest(BaseModel):
    base_url: str
    email: str
    api_token: Optional[str] = None            # omit on edit to keep the stored token
    space_key: str
    requirement_label: str = "requirement"
    qa_label: str = "qa"
    bug_label: str = "bug"


@router.get("/projects")
def list_projects(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Projects this user may query — feeds the project selector (FR-17)."""
    return {"projects": KnowledgeAssistant(db).list_projects(user)}


@router.post("/projects/{project_id}/sync")
def sync_project(project_id: int, db: Session = Depends(get_db),
                 user: User = Depends(get_current_user)):
    """Synchronize the project's documents into the knowledge base."""
    if user.role not in ("super_admin", "org_admin", "pm"):
        raise HTTPException(status_code=403, detail="Not allowed to sync documents.")
    try:
        stats = KnowledgeAssistant(db).sync_project(user, project_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if stats.total_chunks == 0 and (stats.added_pages + stats.updated_pages) == 0 \
            and stats.unchanged_pages == 0:
        raise HTTPException(
            status_code=422,
            detail="No documents found for this project. Configure a Confluence "
                   "space or a local seed folder first.",
        )
    return stats.as_dict()


@router.get("/projects/{project_id}/confluence")
def get_confluence(project_id: int, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    """Return the project's Confluence connection status (never the token)."""
    try:
        conn = KnowledgeAssistant(db).get_confluence_connection(user, project_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return conn or {"connected": False}


@router.post("/projects/{project_id}/confluence")
def set_confluence(project_id: int, body: ConfluenceConnectionRequest,
                   db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Save/update the Confluence connection for a project."""
    if user.role not in ("super_admin", "org_admin", "pm"):
        raise HTTPException(status_code=403, detail="Not allowed to configure documents.")
    labels = {"requirement": body.requirement_label, "qa": body.qa_label, "bug": body.bug_label}
    try:
        KnowledgeAssistant(db).set_confluence_connection(
            user, project_id, base_url=body.base_url, email=body.email,
            api_token=body.api_token, space_key=body.space_key, labels=labels,
        )
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return {"connected": True}


@router.post("/projects/{project_id}/confluence/test")
def test_confluence(project_id: int, db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)):
    """Test the saved Confluence connection (credentials + space reachable)."""
    try:
        return KnowledgeAssistant(db).test_confluence_connection(user, project_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.post("/projects/{project_id}/chat")
def chat(project_id: int, body: ChatRequest, db: Session = Depends(get_db),
         user: User = Depends(get_current_user)):
    """Ask the agentic document assistant a question about the active project."""
    assistant = KnowledgeAssistant(db)
    if project_id not in assistant.allowed_project_ids(user):
        # Denied before any document is accessed (FR-13 acceptance criterion).
        raise HTTPException(status_code=403, detail="Not authorized to access this project.")
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    return assistant.answer(
        user=user, project_id=project_id, question=body.question,
        conversation_id=body.conversation_id, doc_types=body.doc_types,
    )


@router.get("/projects/{project_id}/conversations")
def list_conversations(project_id: int, db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    """List the user's past conversations for a project (newest first)."""
    try:
        convs = KnowledgeAssistant(db).list_conversations(user, project_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return {"conversations": convs}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, db: Session = Depends(get_db),
                        user: User = Depends(get_current_user)):
    """Delete one of the user's conversations."""
    try:
        KnowledgeAssistant(db).delete_conversation(user, conversation_id)
    except AuthorizationError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"deleted": True}


@router.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: int, db: Session = Depends(get_db),
                          user: User = Depends(get_current_user)):
    """Return the message history for one of the user's conversations."""
    conv = db.query(KbConversation).get(conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = (db.query(KbMessage)
                  .filter(KbMessage.conversation_id == conversation_id)
                  .order_by(KbMessage.created_at).all())
    return {"conversation_id": conversation_id, "messages": [
        {"role": m.role, "content": m.content, "citations": m.citations,
         "status": m.status, "created_at": str(m.created_at)}
        for m in messages
    ]}
