"""FR-19 — AI Sprint Planning Agent (requirement-driven).

Generates a sprint-by-sprint plan from a requirement document. Only the
*functional* requirements (the features the software must do) become tasks;
document-meta sections (executive summary, stakeholders, revision history,
assumptions, risk analysis, table of contents, etc.) are ignored. Non-functional
requirements (performance, scalability, security, SEO, …) are NOT standalone
tasks — they are attached as sub-tasks on the relevant functional task.

Each functional task is broken into sub-tasks. Tasks are assigned to developers
by feature affinity, across a PM-specified number of developers, starting with
Sprint 0 (foundation/setup).

Two engines: hosted LLM (Gemini) primary; deterministic heuristic fallback
(always available, used offline / in tests).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import List, Optional

from sqlalchemy.orm import Session

from app.knowledge.config import KnowledgeConfig
from app.knowledge.models import KbDocumentChunk
from app.knowledge.providers.llm import LLMProvider, HostedLLM

# Sections that are NOT build work — never become tasks.
META_KW = ("revision history", "executive summary", "stakeholder", "assumption",
           "dependenc", "risk analysis", "introduction", "glossary", "reference",
           "document control", "approval", "table of contents", "background",
           "conclusion", "scope", "version", "purpose", "overview", "sign-off",
           "change log", "document history", "objective", "document header",
           "proposed solution", "problem statement", "current situation",
           "current system", "goals", "non-goal", "vision")

# Non-functional requirement domains.
NFR_NAMES = ("performance", "scalability", "security", "usability", "reliability",
             "availability", "maintainability", "seo", "accessibility",
             "compliance", "portability", "localization", "observability")
GLOBAL_NFR = ("performance", "scalability", "security", "seo", "accessibility",
              "availability", "reliability")

FOUNDATION_KW = ("architecture", "setup", "migration", "schema", "database",
                 "infrastructure", "ci/cd", "deployment", "scaffold")

THEMES = [
    ("Authentication", ("auth", "sso", "login", "session", "password",
                         "authorization", "access control", "rbac", "role",
                         "permission", "profile", "admin")),
    ("Payments", ("payment", "billing", "invoice", "checkout", "gateway",
                  "transaction", "refund", "pricing")),
    ("Data & Analytics", ("dashboard", "analytics", "report", "reconcil",
                          "search", "filter", "directory", "listing", "taxonomy")),
    ("Content", ("content", "moderation", "article", "page", "publish", "cms")),
    ("Notifications", ("notification", "email", "message", "alert")),
    ("Mobile", ("mobile", "offline", "ios", "android", "app")),
]


# ── Classification & parsing ──────────────────────────────────────────────
def _classify(heading: str) -> str:
    h = heading.lower()
    if "nfr" in h or "non-functional" in h or "non functional" in h:
        return "nfr"
    if h.split(". ")[-1].split()[0:1] and h.split()[0].strip(".0123456789").lower() in NFR_NAMES:
        return "nfr"
    if any(k in h for k in META_KW):
        return "meta"
    if any(k in h for k in FOUNDATION_KW):
        return "foundation"
    return "functional"


def _chunk_features(db: Session, project_id: int, document: Optional[str]) -> List[dict]:
    q = db.query(KbDocumentChunk).filter(
        KbDocumentChunk.project_id == project_id,
        KbDocumentChunk.doc_type == "requirement",
    )
    if document:
        q = q.filter((KbDocumentChunk.page_id == document) |
                     (KbDocumentChunk.page_title == document))
    rows = q.order_by(KbDocumentChunk.id).all()
    features: List[dict] = []
    for r in rows:
        heading = (r.section or "Section").strip()
        if features and features[-1]["heading"] == heading:
            features[-1]["body"].append(r.content)
        else:
            features.append({"heading": heading, "body": [r.content]})
    for f in features:
        f["body"] = " ".join(f["body"]).strip()
    return [f for f in features if f["body"]]


def _split_items(body: str, prefix: str) -> List[dict]:
    """Split a section body into FR-xx / NFR-xx items, if present."""
    parts = re.split(rf"\b({prefix}-\d+)[:.\)]?\s*", body)
    if len(parts) < 3:
        return []
    items = []
    for i in range(1, len(parts), 2):
        marker = parts[i].strip()
        seg = parts[i + 1].strip() if i + 1 < len(parts) else ""
        title = re.split(r"\bdescription\b", seg, flags=re.I)[0].strip()
        title = title.split(".")[0].strip()
        title = " ".join(title.split()[:9]) or marker
        items.append({"name": f"{marker}: {title}", "title": title or marker, "body": seg})
    return items


def _nfr_domain(text: str) -> str:
    t = text.lower()
    for name in NFR_NAMES:
        if name in t:
            return name
    return "quality"


def _parse_text_features(text: str) -> List[dict]:
    """Detect section headings from the raw text (for documents whose headings
    were pasted as plain text and so have no chunk-section labels)."""
    features: List[dict] = []
    current: Optional[dict] = None
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        m = re.match(r"^#{1,6}\s+(.*)$", s)
        heading = None
        if m:
            heading = m.group(1).strip()
        else:
            w = s.split()
            if (len(w) <= 6 and len(s) <= 48 and s[0].isupper()
                    and not s.endswith((".", ":", ",")) and "shall" not in s.lower()
                    and "—" not in s):
                heading = s
        if heading is not None:
            current = {"heading": heading, "body": []}
            features.append(current)
        else:
            if current is None:
                current = {"heading": "Overview", "body": []}
                features.append(current)
            current["body"].append(s)
    for f in features:
        f["body"] = " ".join(f["body"]).strip()
    return [f for f in features if f["body"]]


def _extract(db: Session, project_id: int, document: Optional[str]) -> dict:
    chunk_feats = _chunk_features(db, project_id, document)
    if not chunk_feats:
        return {"functional": [], "foundation": [], "nfrs": []}

    # If chunk sections are meaningful, use them; otherwise the document had no
    # real headings (e.g. titles pasted as plain text) — re-parse from the text.
    generic = {"introduction", "section", "overview"}
    headings = [f["heading"].lower() for f in chunk_feats]
    meaningful = len(set(headings)) > 1 or headings[0] not in generic
    features = chunk_feats if meaningful else \
        _parse_text_features("\n".join(f["body"] for f in chunk_feats))

    functional: List[dict] = []
    foundation: List[dict] = []
    nfrs: List[dict] = []
    meta_pool: List[dict] = []
    for f in features:
        cat = _classify(f["heading"])
        if cat == "meta":
            meta_pool.append(f)
            continue
        if cat == "nfr":
            items = _split_items(f["body"], "NFR")
            if items:
                for it in items:
                    nfrs.append({"label": it["name"], "detail": it["body"][:160],
                                 "domain": _nfr_domain(it["name"] + " " + it["body"])})
            else:
                nfrs.append({"label": f["heading"], "detail": f["body"][:160],
                             "domain": _nfr_domain(f["heading"] + " " + f["body"])})
            continue
        if cat == "foundation":
            foundation.append({"name": f["heading"], "body": f["body"], "section": f["heading"]})
            continue
        # functional — split into FR items when the section lists them.
        items = _split_items(f["body"], "FR")
        if items:
            for it in items:
                functional.append({"name": it["title"], "body": it["body"], "section": it["name"]})
        else:
            functional.append({"name": f["heading"], "body": f["body"], "section": f["heading"]})

    # Safety net: never end up empty when the document has content. If everything
    # classified as meta (e.g. a single "Introduction" blob), treat that content
    # as functional so a plan can still be produced.
    if not functional and not foundation:
        for f in (meta_pool or features):
            items = _split_items(f["body"], "FR")
            if items:
                for it in items:
                    functional.append({"name": it["title"], "body": it["body"], "section": it["name"]})
            else:
                functional.append({"name": f["heading"], "body": f["body"], "section": f["heading"]})

    return {
        "functional": _dedup(functional, "name"),
        "foundation": _dedup(foundation, "name"),
        "nfrs": _dedup(nfrs, "label"),
    }


def _dedup(items: List[dict], key: str) -> List[dict]:
    seen, out = set(), []
    for it in items:
        if it[key] in seen:
            continue
        seen.add(it[key])
        out.append(it)
    return out


def _document_text(extracted: dict) -> str:
    parts = ["# Functional Requirements"]
    for f in extracted["functional"]:
        parts.append(f"## {f['section']}\n{f['body']}")
    if extracted["nfrs"]:
        parts.append("# Non-Functional Requirements")
        for n in extracted["nfrs"]:
            parts.append(f"## {n['label']}\n{n['detail']}")
    return "\n\n".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────
def _theme_of(text: str) -> str:
    t = text.lower()
    for theme, kws in THEMES:
        if any(k in t for k in kws):
            return theme
    return "General"


def _estimate(body: str) -> int:
    return max(8, min(28, len(body) // 45 + 8))


def _applicable_nfrs(req_text: str, nfrs: List[dict]) -> List[dict]:
    t = req_text.lower()
    out, seen = [], set()
    for n in nfrs:
        if n["domain"] in t or any(k in t for k in (n["domain"],)):
            if n["label"] not in seen:
                out.append(n); seen.add(n["label"])
    for n in nfrs:  # always surface cross-cutting NFRs
        if n["domain"] in GLOBAL_NFR and n["label"] not in seen:
            out.append(n); seen.add(n["label"])
    return out[:3]


def _feature_subtasks(base: int) -> List[dict]:
    return [
        {"title": "Design & API contract", "type": "impl", "estimate_hours": max(3, base // 4)},
        {"title": "Backend implementation", "type": "impl", "estimate_hours": max(4, base // 2)},
        {"title": "Frontend / UI", "type": "impl", "estimate_hours": max(3, base // 4)},
        {"title": "Validation & automated tests", "type": "impl", "estimate_hours": max(3, base // 4)},
    ]


def _make_task(name: str, body: str, section: str, dev: str, nfrs: List[dict]) -> dict:
    base = _estimate(body)
    subtasks = _feature_subtasks(base)
    applied = _applicable_nfrs(name + " " + body, nfrs)
    for n in applied:
        subtasks.append({"title": f"Meet {n['label']}", "type": "nfr",
                         "estimate_hours": 2, "detail": n["detail"]})
    total = sum(s["estimate_hours"] for s in subtasks)
    return {"title": name, "description": body[:150], "estimate_hours": total,
            "assigned_developer": dev, "dependencies": [], "source_section": section,
            "subtasks": subtasks, "nfrs": [n["label"] for n in applied]}


# ── Heuristic engine ──────────────────────────────────────────────────────
def _heuristic_plan(extracted: dict, team: List[str], capacity: int,
                    sprint_length_weeks: int, document: str) -> dict:
    functional = extracted["functional"]
    nfrs = extracted["nfrs"]

    # Affinity: assign each theme (in document order) to a developer.
    theme_dev: dict = {}
    order: List[str] = []
    for f in functional:
        t = _theme_of(f["name"] + " " + f["body"])
        if t not in theme_dev:
            theme_dev[t] = team[len(order) % len(team)]
            order.append(t)
    data_dev = theme_dev.get("Data & Analytics", team[min(2, len(team) - 1)])
    auth_dev = theme_dev.get("Authentication", team[0])

    # Sprint 0 — foundation & setup
    sprint0 = [
        _fixed("Project scaffolding & CI/CD pipeline", "Repository, build, and deployment pipeline.", 8, team[0]),
        _fixed("Consolidated API skeleton", "Base API structure shared by all features.", 12, team[0]),
        _fixed("Database schema & migrations", "Initial schema and migration framework.", 10, data_dev),
    ]
    if any(_theme_of(f["name"] + " " + f["body"]) == "Authentication" for f in functional):
        sprint0.append(_fixed("Authentication groundwork", "Shared auth/session foundation used by later features.", 14, auth_dev))
    for f in extracted["foundation"]:
        sprint0.append(_make_task(f["name"], f["body"], f["section"], theme_dev.get(_theme_of(f["name"]), data_dev), nfrs))

    # Functional tasks → later sprints. Themed features keep the same developer
    # (affinity); generic features are spread round-robin to balance the load.
    feature_tasks: List[dict] = []
    generic_i = 0
    for f in functional:
        theme = _theme_of(f["name"] + " " + f["body"])
        if theme == "General":
            dev = team[generic_i % len(team)]
            generic_i += 1
        else:
            dev = theme_dev[theme]
        feature_tasks.append(_make_task(f["name"], f["body"], f["section"], dev, nfrs))

    # Pack per developer into capacity-sized sprint buckets.
    dev_tasks: dict = defaultdict(list)
    for t in feature_tasks:
        dev_tasks[t["assigned_developer"]].append(t)
    buckets: dict = {}
    max_b = 0
    for dev in team:
        bs, cur, acc = [], [], 0
        for t in dev_tasks.get(dev, []):
            if cur and acc + t["estimate_hours"] > capacity:
                bs.append(cur); cur, acc = [], 0
            cur.append(t); acc += t["estimate_hours"]
        if cur:
            bs.append(cur)
        buckets[dev] = bs
        max_b = max(max_b, len(bs))

    sprints = [_sprint("Sprint 0", sprint0)]
    for i in range(max_b):
        tasks = []
        for dev in team:
            if i < len(buckets[dev]):
                tasks.extend(buckets[dev][i])
        sprints.append(_sprint(f"Sprint {i + 1}", tasks))

    return _assemble(document, team, sprint_length_weeks, capacity, sprints, nfrs)


def _fixed(title: str, desc: str, hours: int, dev: str) -> dict:
    return {"title": title, "description": desc, "estimate_hours": hours,
            "assigned_developer": dev, "dependencies": [], "source_section": "Foundation",
            "subtasks": [{"title": title, "type": "impl", "estimate_hours": hours}], "nfrs": []}


def _sprint(name: str, tasks: List[dict]) -> dict:
    totals: dict = defaultdict(int)
    for t in tasks:
        totals[t["assigned_developer"]] += t["estimate_hours"]
    return {"name": name, "tasks": tasks, "total_by_developer": dict(totals)}


def _assemble(document, team, weeks, capacity, sprints, nfrs) -> dict:
    summary: dict = defaultdict(lambda: {"total_hours": 0, "areas": set()})
    for sp in sprints:
        for t in sp["tasks"]:
            d = summary[t["assigned_developer"]]
            d["total_hours"] += t["estimate_hours"]
            d["areas"].add(t["source_section"])
    developer_summary = [
        {"developer": dev, "total_hours": summary[dev]["total_hours"],
         "areas": sorted(summary[dev]["areas"])}
        for dev in team if dev in summary
    ]
    return {
        "source_document": document or "requirement",
        "developer_count": len(team), "team": team,
        "sprint_length_weeks": weeks, "hours_per_developer_per_sprint": capacity,
        "sprints": sprints, "developer_summary": developer_summary,
        "non_functional_requirements": [{"label": n["label"], "detail": n["detail"]} for n in nfrs],
        "assumptions": [
            f"Capacity = {len(team)} developer(s) x {capacity}h per {weeks}-week sprint.",
            "Only functional requirements were turned into tasks; meta sections were ignored.",
            "Non-functional requirements are attached as sub-tasks on the relevant tasks.",
        ],
    }


# ── LLM engine ────────────────────────────────────────────────────────────
def _llm_plan(llm: HostedLLM, doc_text: str, team: List[str], capacity: int,
              weeks: int, document: str, nfrs: List[dict]) -> Optional[dict]:
    schema = ('{"sprints":[{"name":"Sprint 0","tasks":[{"title":"...","description":"...",'
              '"assigned_developer":"dev1","source_section":"...","subtasks":['
              '{"title":"Backend implementation","type":"impl","estimate_hours":6},'
              '{"title":"Meet NFR-01 Performance","type":"nfr","estimate_hours":2}]}]}]}')
    system = (
        "You are an expert software delivery lead. Produce a sprint plan as STRICT JSON "
        "only (no prose, no code fences). Rules: (1) Use ONLY the functional requirements "
        "— the features the software must do. IGNORE non-build sections such as executive "
        "summary, stakeholders, revision history, table of contents, assumptions, "
        "dependencies, risk analysis, glossary and references. (2) Create tasks for EVERY "
        "functional requirement in the document (e.g. FR-00, FR-01, FR-02 … through the "
        "last one) — do not stop after the first. (3) Non-functional requirements "
        "(performance, scalability, security, SEO, accessibility, usability) must NOT be "
        "standalone tasks; attach them as sub-tasks of type 'nfr' on the functional task "
        "they apply to. (4) Break every functional requirement into implementation "
        "sub-tasks (type 'impl'). (5) Begin with 'Sprint 0' for foundational/setup tasks, "
        "then use as many sprints as needed to cover all requirements. (6) Assign tasks "
        f"across exactly {team}; group related tasks under the same developer; no developer "
        f"exceeds {capacity} hours per sprint. (7) Every task references its requirement "
        "section in 'source_section'."
    )
    prompt = (f"Requirement document:\n{doc_text[:24000]}\n\nDevelopers: {team}\n"
              f"Sprint length: {weeks} weeks\nHours per developer per sprint: {capacity}\n\n"
              f"Plan ALL functional requirements found above. "
              f"Return JSON exactly in this shape: {schema}")
    try:
        raw = llm._complete(system, prompt, max_tokens=16000, response_json=True)
    except Exception:
        return None
    data = _parse_json(raw)
    if not data or not isinstance(data.get("sprints"), list) or not data["sprints"]:
        return None
    sprints = []
    for sp in data["sprints"]:
        tasks = []
        for t in sp.get("tasks", []):
            subs = []
            for s in (t.get("subtasks") or []):
                subs.append({"title": str(s.get("title", "Subtask")),
                             "type": "nfr" if s.get("type") == "nfr" else "impl",
                             "estimate_hours": int(s.get("estimate_hours", 4) or 4),
                             "detail": str(s.get("detail", ""))})
            if not subs:
                subs = [{"title": "Implementation", "type": "impl", "estimate_hours": 8}]
            total = sum(s["estimate_hours"] for s in subs)
            tasks.append({
                "title": str(t.get("title", "Task")), "description": str(t.get("description", "")),
                "estimate_hours": total,
                "assigned_developer": t.get("assigned_developer") if t.get("assigned_developer") in team else team[0],
                "dependencies": t.get("dependencies", []) or [], "source_section": str(t.get("source_section", "")),
                "subtasks": subs, "nfrs": [s["title"] for s in subs if s["type"] == "nfr"]})
        sprints.append(_sprint(str(sp.get("name", "Sprint")), tasks))
    return _assemble(document, team, weeks, capacity, sprints, nfrs)


def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ── Public API ────────────────────────────────────────────────────────────
class SprintPlanner:
    def __init__(self, db: Session, llm: LLMProvider, config: KnowledgeConfig):
        self.db = db
        self.llm = llm
        self.config = config

    def plan(self, project_id: int, document: Optional[str], developer_count: int,
             sprint_length_weeks: int, hours_per_developer: int) -> dict:
        developer_count = max(1, min(10, int(developer_count)))
        team = [f"dev{i + 1}" for i in range(developer_count)]
        capacity = max(10, int(hours_per_developer))

        extracted = _extract(self.db, project_id, document)
        if not extracted["functional"] and not extracted["foundation"]:
            raise ValueError("No functional requirements found in this document. "
                             "Sync a requirement document with a Functional Requirements section.")
        doc_text = _document_text(extracted)

        if isinstance(self.llm, HostedLLM):
            result = _llm_plan(self.llm, doc_text, team, capacity, sprint_length_weeks,
                               document or "requirement", extracted["nfrs"])
            if result:
                result["engine"] = "llm"
                return result

        result = _heuristic_plan(extracted, team, capacity, sprint_length_weeks,
                                 document or "requirement")
        result["engine"] = "heuristic"
        return result
