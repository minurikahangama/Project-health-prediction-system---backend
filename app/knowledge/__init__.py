"""
FR-13 — Project Knowledge Base and Agentic Document Assistant.

RAG over Confluence (Requirement / QA / Bug-report documents), answered by a
LangGraph-orchestrated agent that authorizes, retrieves, self-corrects,
verifies grounding, and cites sources. See FR-13_Implementation_Plan.md.

Provider selection is environment-driven so the feature runs fully offline for
tests (deterministic hashing embeddings + a local extractive LLM + a portable
JSON vector index) and against Confluence + a hosted LLM in production.
"""
