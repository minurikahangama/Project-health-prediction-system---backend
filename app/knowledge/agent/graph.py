"""LangGraph wiring for the agentic document assistant (FR-13).

Flow:
    authorize → grade_question → retrieve → grade_docs
        grade_docs: relevant → generate ; weak & retries left → rewrite → retrieve
                    weak & no retries → not_available
        generate → check_grounding: grounded → respond ; else → not_available
Any authorization/scope failure short-circuits to a terminal node.
Every node appends to `trace` for the reasoning log.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.knowledge.agent.state import ChatState
from app.knowledge.config import KnowledgeConfig
from app.knowledge.providers.llm import LLMProvider
from app.knowledge.providers.web_search import WebSearchProvider
from app.knowledge.vector_index import VectorIndex

NOT_AVAILABLE_MSG = "I could not find this information in the project's documents."
DENIED_MSG = "You are not authorized to access this project."
OUT_OF_SCOPE_MSG = "I can only answer questions about this project's documentation."
WEB_DISABLED_MSG = "Web search is not configured. Add a TAVILY_API_KEY to enable it."
WEB_EMPTY_MSG = "I couldn't find anything useful on the web for that question."


class AgentRuntime:
    """Holds the retriever + LLM and exposes each graph node as a method."""

    def __init__(self, retriever: VectorIndex, llm: LLMProvider, config: KnowledgeConfig,
                 web: WebSearchProvider | None = None):
        self.retriever = retriever
        self.llm = llm
        self.config = config
        self.web = web or WebSearchProvider()

    # ── Nodes ────────────────────────────────────────────────────────────
    def authorize(self, state: ChatState) -> dict:
        allowed = state.get("allowed_project_ids", [])
        ok = state["active_project_id"] in allowed
        return {"authorized": ok,
                "trace": [{"step": "authorize", "detail": {"authorized": ok}}]}

    def deny(self, state: ChatState) -> dict:
        return {"answer": DENIED_MSG, "citations": [], "status": "denied",
                "trace": [{"step": "deny", "detail": {}}]}

    def grade_question(self, state: ChatState) -> dict:
        in_scope = self.llm.classify_in_scope(state["question"])
        return {"in_scope": in_scope,
                "trace": [{"step": "grade_question", "detail": {"in_scope": in_scope}}]}

    def out_of_scope(self, state: ChatState) -> dict:
        return {"answer": OUT_OF_SCOPE_MSG, "citations": [], "status": "out_of_scope",
                "trace": [{"step": "out_of_scope", "detail": {}}]}

    def retrieve(self, state: ChatState) -> dict:
        passages = self.retriever.search(
            project_id=state["active_project_id"],
            query=state["question"],
            top_k=self.config.top_k,
            doc_types=state.get("doc_types"),
        )
        best = passages[0].score if passages else 0.0
        return {
            "passages": [p.as_dict() for p in passages],
            "best_score": best,
            "trace": [{"step": "retrieve", "detail": {
                "count": len(passages), "best_score": round(best, 4),
                "query": state["question"]}}],
        }

    def grade_docs(self, state: ChatState) -> dict:
        relevant = state.get("best_score", 0.0) >= self.config.relevance_min
        return {"trace": [{"step": "grade_docs", "detail": {
            "relevant": relevant, "best_score": round(state.get("best_score", 0.0), 4),
            "threshold": self.config.relevance_min}}]}

    def rewrite_query(self, state: ChatState) -> dict:
        attempts = state.get("attempts", 0) + 1
        new_q = self.llm.rewrite_query(state.get("original_question", state["question"]))
        return {"question": new_q, "attempts": attempts,
                "trace": [{"step": "rewrite_query", "detail": {
                    "attempt": attempts, "rewritten": new_q}}]}

    def generate(self, state: ChatState) -> dict:
        passages = state["passages"]
        result = self.llm.generate(state.get("original_question", state["question"]), passages)
        used = result.get("used", [])
        citations = [{
            "page_id": passages[i]["page_id"],
            "page_title": passages[i].get("page_title"),
            "section": passages[i].get("section"),
            "doc_type": passages[i]["doc_type"],
        } for i in used if i < len(passages)]
        return {"answer": result["answer"], "citations": citations,
                "trace": [{"step": "generate", "detail": {"citations": len(citations)}}]}

    def check_grounding(self, state: ChatState) -> dict:
        grounded = self.llm.check_grounding(state.get("answer", ""), state["passages"])
        return {"grounded": grounded,
                "trace": [{"step": "check_grounding", "detail": {"grounded": grounded}}]}

    def respond(self, state: ChatState) -> dict:
        return {"status": "answered",
                "trace": [{"step": "respond", "detail": {"status": "answered"}}]}

    def not_available(self, state: ChatState) -> dict:
        return {"answer": NOT_AVAILABLE_MSG, "citations": [], "status": "not_available",
                "trace": [{"step": "not_available", "detail": {}}]}

    def web_search(self, state: ChatState) -> dict:
        """User toggled 'Search the web'. Pull light project context for grounding,
        run a Tavily search, and synthesise an implementation-oriented answer."""
        question = state.get("original_question", state["question"])
        if not self.web.enabled:
            return {"answer": WEB_DISABLED_MSG, "citations": [], "status": "not_available",
                    "trace": [{"step": "web_search", "detail": {"enabled": False}}]}

        # Best-effort project context (no relevance gate — it's only for grounding).
        doc_passages = [p.as_dict() for p in self.retriever.search(
            project_id=state["active_project_id"], query=question,
            top_k=self.config.web_context_top_k, doc_types=state.get("doc_types"))]
        results = self.web.search(question, max_results=self.config.web_max_results)
        if not results:
            return {"answer": WEB_EMPTY_MSG, "citations": [], "status": "not_available",
                    "trace": [{"step": "web_search",
                               "detail": {"enabled": True, "web_results": 0}}]}

        web_dicts = [r.as_dict() for r in results]
        gen = self.llm.generate_web(question, web_dicts, doc_passages)
        citations = [{
            "page_id": web_dicts[i]["url"], "page_title": web_dicts[i]["title"],
            "section": None, "doc_type": "web", "url": web_dicts[i]["url"],
        } for i in gen.get("used_web", []) if i < len(web_dicts)]
        citations += [{
            "page_id": doc_passages[i]["page_id"],
            "page_title": doc_passages[i].get("page_title"),
            "section": doc_passages[i].get("section"),
            "doc_type": doc_passages[i]["doc_type"],
        } for i in gen.get("used_docs", []) if i < len(doc_passages)]
        return {
            "answer": gen.get("answer", "") or WEB_EMPTY_MSG,
            "citations": citations,
            "status": "web_answered" if gen.get("answer") else "not_available",
            "web_results": web_dicts,
            "trace": [{"step": "web_search", "detail": {
                "enabled": True, "web_results": len(web_dicts),
                "doc_context": len(doc_passages)}}],
        }

    # ── Routers ──────────────────────────────────────────────────────────
    def _route_after_authorize(self, state: ChatState) -> str:
        if not state.get("authorized"):
            return "deny"
        if state.get("web_search"):
            return "web_answer"
        return "grade_question"

    def _route_after_grade_question(self, state: ChatState) -> str:
        return "retrieve" if state.get("in_scope") else "out_of_scope"

    def _route_after_grade_docs(self, state: ChatState) -> str:
        if state.get("best_score", 0.0) >= self.config.relevance_min:
            return "generate"
        if state.get("attempts", 0) < self.config.max_retries:
            return "rewrite_query"
        return "not_available"

    def _route_after_grounding(self, state: ChatState) -> str:
        return "respond" if state.get("grounded") else "not_available"


def build_agent(runtime: AgentRuntime):
    """Compile the LangGraph app for the given runtime."""
    g = StateGraph(ChatState)

    g.add_node("authorize", runtime.authorize)
    g.add_node("deny", runtime.deny)
    g.add_node("grade_question", runtime.grade_question)
    g.add_node("out_of_scope", runtime.out_of_scope)
    g.add_node("retrieve", runtime.retrieve)
    g.add_node("grade_docs", runtime.grade_docs)
    g.add_node("rewrite_query", runtime.rewrite_query)
    g.add_node("generate", runtime.generate)
    g.add_node("check_grounding", runtime.check_grounding)
    g.add_node("respond", runtime.respond)
    g.add_node("not_available", runtime.not_available)
    g.add_node("web_answer", runtime.web_search)

    g.add_edge(START, "authorize")
    g.add_conditional_edges("authorize", runtime._route_after_authorize,
                            {"grade_question": "grade_question", "web_answer": "web_answer",
                             "deny": "deny"})
    g.add_edge("web_answer", END)
    g.add_edge("deny", END)
    g.add_conditional_edges("grade_question", runtime._route_after_grade_question,
                            {"retrieve": "retrieve", "out_of_scope": "out_of_scope"})
    g.add_edge("out_of_scope", END)
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges("grade_docs", runtime._route_after_grade_docs,
                            {"generate": "generate", "rewrite_query": "rewrite_query",
                             "not_available": "not_available"})
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "check_grounding")
    g.add_conditional_edges("check_grounding", runtime._route_after_grounding,
                            {"respond": "respond", "not_available": "not_available"})
    g.add_edge("respond", END)
    g.add_edge("not_available", END)

    return g.compile()
