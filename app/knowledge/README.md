# FR-13 — Agentic Document Assistant (RAG over Confluence with LangGraph)

Implements FR-13 from `FR-13_Implementation_Plan.md`: a per-project knowledge
base plus a LangGraph agent that answers questions grounded in the project's
Requirement / QA / Bug documents, with citations, self-correcting retrieval, a
grounding check, and full project isolation.

## Design

- **Offline by default.** Providers are pluggable and default to an offline path
  (deterministic hashing embeddings + a local extractive LLM + a portable JSON
  vector index), so the feature runs and is fully tested without any API key.
  Set `KB_*` in `.env` to switch to Confluence + sentence-transformers + Claude.
- **Project isolation** is structural: every chunk is stamped with `project_id`
  and every retrieval filters on it, so no query can reach another project's data.
- **The agent** is a LangGraph state machine:
  `authorize → grade_question → retrieve → grade_docs → (rewrite ↻ retrieve | generate) → check_grounding → respond | not_available`.

## Layout

```
app/knowledge/
  config.py            KB_* settings (offline defaults)
  models.py            KB tables: chunks, conversations, messages, reasoning trace
  chunker.py           section-aware passage splitting
  vector_index.py      project-scoped cosine retrieval
  ingestion.py         incremental sync (add/update/skip/delete)
  providers/           embeddings (hashing | sentence-transformers), llm (local | anthropic)
  sources/             confluence (prod) + local folder (offline/seed)
  agent/               state.py, graph.py (LangGraph nodes + wiring)
  service.py           KnowledgeAssistant — auth + persistence + agent
  seed_docs/           sample Alpha/Beta projects for the demo/tests
app/api/knowledge.py   REST endpoints
tests/test_knowledge_assistant.py
scripts/knowledge_demo.py
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET  | `/knowledge/projects` | list projects the user may query (selector, FR-17) |
| POST | `/knowledge/projects/{id}/sync` | sync documents into the knowledge base |
| POST | `/knowledge/projects/{id}/chat` | ask the agentic chatbot |
| GET  | `/knowledge/conversations/{id}/messages` | conversation history |

`chat` body: `{ "question": "...", "conversation_id": 12, "doc_types": ["requirement"] }`
returns `{ answer, citations, status, trace, conversation_id, message_id }`.

## Run the demo (no external services)

```bash
cd phps-research
DATABASE_URL="sqlite:///./_kb_demo.db" python scripts/knowledge_demo.py
```

## Run the tests

```bash
cd phps-research
python -m unittest tests.test_knowledge_assistant -v
```

Covers: incremental sync, grounded+cited answers, project isolation, out-of-domain
"not available", authorization denial, multi-turn memory, and reasoning-trace logging.

## Switch to production providers

In `.env`:

```
KB_EMBEDDING_PROVIDER=sentence-transformers
KB_LLM_PROVIDER=gemini
KB_LLM_MODEL=gemini-2.0-flash
GEMINI_API_KEY=your-gemini-api-key
KB_DEFAULT_SOURCE=confluence
```

Supported LLM providers: `local` (offline default), `gemini` (Google Gemini via
the `google-genai` SDK), `anthropic` (Claude). The provider is chosen entirely
by `KB_LLM_PROVIDER`, so no code changes are needed to switch.

and register each project's Confluence space via `kb_project_sources`
(`source_type='confluence'`, `space_key`, `doc_type`, and a `config` JSON holding
`base_url` / `email` / `api_token` / `label`). For large corpora, replace the
JSON embedding column + Python cosine in `vector_index.py` with a native
`pgvector` column and an `ORDER BY embedding <=> :q` query.
