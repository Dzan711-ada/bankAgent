from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from bank_agent.config import Settings
from bank_agent.providers import serialize_evidence
from bank_agent.schemas import AnswerDraft, Clause, EvidenceView, QueryRequest, QueryResponse, UserContext


class QuestionAnalysis(BaseModel):
    """User facts and retrieval planning; this is not a policy authority."""

    model_config = ConfigDict(extra="forbid")
    resolved_question: str = Field(min_length=1, max_length=8000)
    facts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    subquestions: list[str] = Field(default_factory=list)
    complexity: Literal["simple", "complex"] = "simple"
    needs_clarification: bool = False


class AgentState(TypedDict, total=False):
    request: QueryRequest
    user: UserContext
    session_id: str
    trace_id: str
    started_at: float
    snapshot_id: str
    allowed: dict[str, Clause]
    history: list[dict]
    analysis: QuestionAnalysis
    evidence: list[Clause]
    seen_ids: set[str]
    dependency_ids: set[str]
    supplement_rounds: int
    subquestions_used: int
    repair_count: int
    draft: AnswerDraft
    validation_errors: list[str]
    checks: list[str]
    trace: list[str]
    model_steps: list[dict]
    retrieval_queries: list[dict]
    response: QueryResponse


@dataclass
class _Session:
    updated_at: float
    history: list[dict]


class BankPolicyAgent:
    """Bounded, permission-aware LangGraph policy consultation workflow.

    Session memory contains user inputs only, with a 30-minute TTL, at most 16
    messages per session and 500 sessions. A session is also bound to the user's
    identity, role and organization. This local MVP memory is not durable.
    """

    SESSION_TTL_SECONDS = 1800
    MAX_SESSIONS = 500
    MAX_HISTORY_MESSAGES = 16
    MAX_DEPENDENCY_LOOKUPS = 64
    PROMPT_RESERVE_TOKENS = 1200

    def __init__(self, settings: Settings, store, retriever, provider):
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.provider = provider
        self._sessions: dict[tuple[str, str, str, str], _Session] = {}
        self._memory_lock = threading.RLock()
        self._last_state: AgentState | None = None
        # Preserve ordering within a conversation and avoid interleaved provider
        # accounting for providers that do not use thread-local usage records.
        self._request_lock = threading.RLock()
        builder = StateGraph(AgentState)
        for name, node in [
            ("access", self._access),
            ("analyze", self._analyze),
            ("retrieve", self._retrieve),
            ("supplement", self._supplement),
            ("generate", self._generate),
            ("validate", self._validate),
            ("repair", self._repair),
            ("finalize", self._finalize),
        ]:
            builder.add_node(name, self._instrument_node(name, node))
        builder.add_edge(START, "access")
        builder.add_conditional_edges("access", self._after_access, {"analyze": "analyze", "finalize": "finalize"})
        builder.add_conditional_edges("analyze", self._after_analysis,
                                      {"retrieve": "retrieve", "finalize": "finalize"})
        builder.add_edge("retrieve", "supplement")
        builder.add_conditional_edges("supplement", self._after_supplement,
                                      {"supplement": "supplement", "generate": "generate"})
        builder.add_edge("generate", "validate")
        builder.add_conditional_edges("validate", self._after_validation,
                                      {"repair": "repair", "finalize": "finalize"})
        builder.add_edge("repair", "validate")
        builder.add_edge("finalize", END)
        self.graph = builder.compile()

    def ask(self, request: QueryRequest, user: UserContext) -> QueryResponse:
        session_id = request.session_id or uuid.uuid4().hex
        key = (user.user_id, user.role, user.organization, session_id)
        with self._request_lock:
            history = self._read_history(key, request)
            state: AgentState = {
                "request": request, "user": user, "session_id": session_id,
                "trace_id": uuid.uuid4().hex, "started_at": time.perf_counter(),
                "snapshot_id": "", "history": history, "evidence": [],
                "seen_ids": set(), "dependency_ids": set(), "supplement_rounds": 0,
                "subquestions_used": 0, "repair_count": 0, "checks": [],
                "trace": [], "model_steps": [], "validation_errors": [],
                "retrieval_queries": [],
            }
            self._last_state = state
            try:
                self.provider.begin_request()
                result = self.graph.invoke(state, config={"recursion_limit": 24})
                response = result["response"]
            except Exception:
                # Internal exceptions may contain document content or credentials.
                # A failed access, retrieval or model call must never release a draft.
                failed_state = self._last_state or state
                response = QueryResponse(
                    status="review_required", answer="本次查询未能完成校验，请稍后重试或联系制度管理员。",
                    snapshot_id=failed_state["snapshot_id"], session_id=session_id,
                    trace_id=state["trace_id"], checks=["workflow_failed_closed"],
                    demo_mode=self.settings.demo_mode,
                )
                try:
                    self._audit(failed_state, response)
                except Exception:
                    response = response.model_copy(update={"checks": ["workflow_failed_closed", "audit_unavailable"]})
            self._remember(key, request.question)
            self._last_state = None
            return response

    def _instrument_node(self, name: str, operation):
        def run(state: AgentState):
            self._last_state = state
            try:
                return operation(state)
            except Exception:
                state["trace"].append(name + ":failed")
                raise
        return run

    def _count(self, text: str) -> int:
        return max(0, int(self.provider.count_tokens(text)))

    def _read_history(self, key: tuple[str, str, str, str], request: QueryRequest) -> list[dict]:
        now = time.monotonic()
        with self._memory_lock:
            self._sessions = {k: v for k, v in self._sessions.items()
                              if now - v.updated_at < self.SESSION_TTL_SECONDS}
            session = self._sessions.get(key)
            candidates = list(session.history) if session else []
        reserve = self._count(json.dumps(QuestionAnalysis.model_json_schema(), ensure_ascii=False))
        budget = self.settings.max_context_tokens - reserve - self.PROMPT_RESERVE_TOKENS
        spent = self._count(request.question)
        selected = []
        for item in reversed(candidates):
            cost = self._count(json.dumps(item, ensure_ascii=False))
            if spent + cost > budget:
                break
            selected.append(dict(item))
            spent += cost
        return list(reversed(selected))

    def _remember(self, key: tuple[str, str, str, str], question: str) -> None:
        with self._memory_lock:
            session = self._sessions.get(key, _Session(time.monotonic(), []))
            session.history.append({"role": "user", "content": question})
            session.history = session.history[-self.MAX_HISTORY_MESSAGES:]
            session.updated_at = time.monotonic()
            self._sessions[key] = session
            while len(self._sessions) > self.MAX_SESSIONS:
                oldest = min(self._sessions, key=lambda candidate: self._sessions[candidate].updated_at)
                del self._sessions[oldest]

    @staticmethod
    def _permitted(clause: Clause, user: UserContext, request: QueryRequest) -> bool:
        metadata = clause.metadata
        return (
            metadata.reviewed
            and metadata.effective_date <= request.business_date
            and (metadata.expiry_date is None or request.business_date < metadata.expiry_date)
            and user.role in metadata.allowed_roles
            and (metadata.access_level != "restricted" or user.role in {"compliance", "admin"})
            and (not metadata.organizations or user.organization in metadata.organizations)
            and (not request.scope or not metadata.scopes or request.scope in metadata.scopes)
            and (metadata.document_type != "teaching" or request.include_teaching)
        )

    def _access(self, state: AgentState) -> dict:
        snapshot_id = self.store.active_snapshot()
        allowed = {clause.evidence_id: clause for clause in
                   self.store.allowed_clauses(state["user"], state["request"], snapshot_id=snapshot_id)
                   if self._permitted(clause, state["user"], state["request"])} if snapshot_id else {}
        updates = {"snapshot_id": snapshot_id, "allowed": allowed,
                   "trace": [*state["trace"], "access"], "checks": [*state["checks"], "access_date_version_checked"]}
        if not allowed:
            updates["draft"] = AnswerDraft(
                status="insufficient_evidence",
                answer="当前没有您有权访问且在指定业务日期有效的已审核制度。请先导入并审核适用制度。",
            )
        return updates

    @staticmethod
    def _after_access(state: AgentState) -> str:
        return "analyze" if state["allowed"] else "finalize"

    def _model_call(self, state: AgentState, name: str, operation):
        started = time.perf_counter()
        try:
            result = operation()
            succeeded = True
            return result
        finally:
            state["model_steps"].append({"step": name, "succeeded": locals().get("succeeded", False),
                                         "duration_ms": round((time.perf_counter() - started) * 1000, 2)})

    def _analyze(self, state: AgentState) -> dict:
        request = state["request"]
        payload = json.dumps({"question": request.question, "user_history": state["history"]}, ensure_ascii=False)
        reserve = self._count(json.dumps(QuestionAnalysis.model_json_schema(), ensure_ascii=False))
        if self._count(payload) + reserve + self.PROMPT_RESERVE_TOKENS > self.settings.max_context_tokens:
            return {"draft": AnswerDraft(status="clarification_needed", answer="问题超过当前上下文预算，请缩短问题并分次补充事实。",
                                          missing_information=["请缩短问题，保留业务对象、操作及关键事实。"]),
                    "trace": [*state["trace"], "analyze"], "checks": [*state["checks"], "question_budget_exceeded"]}
        analysis = self._model_call(state, "analyze", lambda: self.provider.analyze(request.question, state["history"]))
        if not isinstance(analysis, QuestionAnalysis):
            analysis = QuestionAnalysis.model_validate(analysis)
        # Planning output cannot expand the configured work budget.
        analysis = analysis.model_copy(update={
            "facts": [item[:2000] for item in analysis.facts[:12]],
            "missing_information": [item[:500] for item in analysis.missing_information[:8]],
            "subquestions": list(dict.fromkeys(item.strip()[:1000] for item in analysis.subquestions if item.strip()))[
                :self.settings.max_subquestions],
        })
        updates = {"analysis": analysis, "trace": [*state["trace"], "analyze"]}
        if analysis.needs_clarification:
            missing = analysis.missing_information or ["请补充业务对象、拟执行的操作和关键业务事实。"]
            updates["draft"] = AnswerDraft(status="clarification_needed", answer="请补充以下业务事实后再查询。",
                                           missing_information=missing)
        return updates

    @staticmethod
    def _after_analysis(state: AgentState) -> str:
        return "finalize" if "draft" in state else "retrieve"

    def _fit_evidence(self, state: AgentState, candidates: list[Clause]) -> list[Clause]:
        schema_cost = self._count(json.dumps(AnswerDraft.model_json_schema(), ensure_ascii=False))
        selected: list[Clause] = []
        for clause in candidates:
            if len(selected) >= self.settings.evidence_k:
                break
            payload = {"question": state["request"].question, "analysis": state["analysis"].model_dump(),
                       "evidence": serialize_evidence([*selected, clause]),
                       "repair_errors": []}
            # Reserve space for the system prompt and one bounded repair's errors.
            cost = self._count(json.dumps(payload, ensure_ascii=False)) + schema_cost + self.PROMPT_RESERVE_TOKENS + 512
            if cost <= self.settings.max_context_tokens:
                selected.append(clause)
        return selected

    def _merge_hits(self, state: AgentState, hits) -> list[Clause]:
        selected = list(state["evidence"])
        ids = {clause.evidence_id for clause in selected}
        for hit in hits[:self.settings.retrieval_k]:
            evidence_id = hit.clause.evidence_id
            state["seen_ids"].add(evidence_id)
            # Use the canonical snapshot clause, never text supplied by a retriever.
            canonical = state["allowed"].get(evidence_id)
            if canonical is not None and evidence_id not in ids:
                selected.append(canonical)
                ids.add(evidence_id)
        return self._fit_evidence(state, selected)

    def _retrieve(self, state: AgentState) -> dict:
        query = state["analysis"].resolved_question
        hits = self.retriever.search(query, state["user"], state["request"],
                                     snapshot_id=state["snapshot_id"])
        self._record_retrieval(state, query, hits)
        return {"evidence": self._merge_hits(state, hits), "trace": [*state["trace"], "retrieve"],
                "checks": [*state["checks"], "retrieval_access_rechecked", "context_budget_checked"]}

    @staticmethod
    def _record_retrieval(state: AgentState, query: str, hits) -> None:
        # Audit sees all authorized retrieval hits, including those omitted by
        # the context budget. Unauthorized document IDs are never recorded.
        state["retrieval_queries"].append({
            "query": query,
            "evidence_ids": list(dict.fromkeys(hit.clause.evidence_id for hit in hits
                                              if hit.clause.evidence_id in state["allowed"])),
        })

    def _supplement(self, state: AgentState) -> dict:
        maximum = self.settings.max_supplement_rounds
        if state["supplement_rounds"] >= maximum:
            return {"trace": [*state["trace"], "supplement"]}
        selected = list(state["evidence"])
        if state["supplement_rounds"] == 0:
            for query in state["analysis"].subquestions[:self.settings.max_subquestions]:
                if state["subquestions_used"] >= self.settings.max_subquestions:
                    break
                hits = self.retriever.search(query, state["user"], state["request"], snapshot_id=state["snapshot_id"])
                self._record_retrieval(state, query, hits)
                state["evidence"] = selected
                selected = self._merge_hits(state, hits)
                state["subquestions_used"] += 1
        ids = {clause.evidence_id for clause in selected}
        # Read related dependencies through the access-filtered store API again.
        # Two rounds permit two dependency hops, with a separate fanout ceiling.
        for clause in list(selected):
            for related_id in clause.related_ids:
                if len(state["dependency_ids"]) >= self.MAX_DEPENDENCY_LOOKUPS:
                    break
                if related_id in state["dependency_ids"] or related_id in ids:
                    continue
                state["dependency_ids"].add(related_id)
                related = self.store.get_clause(related_id, state["user"], state["request"],
                                                snapshot_id=state["snapshot_id"])
                canonical = state["allowed"].get(related_id)
                if related is not None and canonical is not None and self._permitted(related, state["user"], state["request"]):
                    selected.append(canonical)
                    ids.add(related_id)
        return {"evidence": self._fit_evidence(state, selected),
                "supplement_rounds": state["supplement_rounds"] + 1,
                "subquestions_used": state["subquestions_used"],
                "trace": [*state["trace"], "supplement"],
                "checks": [*state["checks"], "supplement_bounded", "dependency_access_rechecked"]}

    def _after_supplement(self, state: AgentState) -> str:
        return "supplement" if state["supplement_rounds"] < self.settings.max_supplement_rounds else "generate"

    def _generate(self, state: AgentState) -> dict:
        if not state["evidence"]:
            draft = AnswerDraft(status="insufficient_evidence", answer="未检索到可用于本问题的有效证据，无法给出制度结论。")
        else:
            draft = self._model_call(state, "generate", lambda: self.provider.generate(
                state["request"].question, state["analysis"], state["evidence"]))
            if not isinstance(draft, AnswerDraft):
                draft = AnswerDraft.model_validate(draft)
        return {"draft": draft, "trace": [*state["trace"], "generate"]}

    def _program_errors(self, state: AgentState) -> list[str]:
        draft = state["draft"]
        evidence = {clause.evidence_id: clause for clause in state["evidence"]}
        errors = []
        if draft.claims and any(related_id not in evidence for clause in evidence.values()
                                for related_id in clause.related_ids):
            errors.append("referenced_dependency_unavailable_within_budget")
        if draft.status == "answered" and not draft.claims:
            errors.append("answered_requires_cited_claims")
        for index, claim in enumerate(draft.claims):
            prefix = f"claim_{index + 1}"
            if not claim.evidence_ids or len(claim.evidence_ids) != len(claim.supporting_quotes):
                errors.append(prefix + "_citation_quote_pairs_required")
                continue
            if len(set(claim.evidence_ids)) != len(claim.evidence_ids):
                errors.append(prefix + "_duplicate_citation")
            for evidence_id, quote in zip(claim.evidence_ids, claim.supporting_quotes):
                clause = evidence.get(evidence_id)
                if clause is None or evidence_id not in state["allowed"]:
                    errors.append(prefix + "_invalid_evidence_id")
                elif not self._permitted(clause, state["user"], state["request"]):
                    errors.append(prefix + "_access_or_date_invalid")
                elif not quote.strip() or quote not in clause.text:
                    errors.append(prefix + "_quote_not_exact_original")
                elif not self._complete_excerpt(quote, clause.text):
                    errors.append(prefix + "_quote_must_preserve_statement_boundaries")
            # Exact extraction prevents changing dates, numbers and negation and
            # prevents an unverified paraphrase from masquerading as a citation.
            if claim.text != "\n".join(claim.supporting_quotes):
                errors.append(prefix + "_claim_must_equal_exact_quotes")
        for index, condition in enumerate(draft.applicable_conditions):
            if not condition.strip() or not any(condition in clause.text and self._complete_excerpt(condition, clause.text)
                                                for clause in state["evidence"]):
                errors.append(f"condition_{index + 1}_not_exact_original")
        return errors

    @staticmethod
    def _complete_excerpt(quote: str, original: str) -> bool:
        """Reject deleting a negation, a digit, or a condition from inside a sentence."""
        if quote == original:
            return True
        boundaries = "\n。！？；.!?;"
        start = original.find(quote)
        while start >= 0:
            end = start + len(quote)
            left = start == 0 or original[start - 1] in boundaries
            right = end == len(original) or original[end] in boundaries or quote[-1] in boundaries
            if left and right:
                return True
            start = original.find(quote, start + 1)
        return False

    def _validate(self, state: AgentState) -> dict:
        errors = self._program_errors(state)
        if state["draft"].claims and state["evidence"]:
            review = self._model_call(state, "verify", lambda: self.provider.verify(
                state["request"].question, state["draft"], state["evidence"]))
            errors.extend(str(error)[:160] for error in review[:8])
        return {"validation_errors": list(dict.fromkeys(errors)),
                "trace": [*state["trace"], "validate"],
                "checks": [*state["checks"], "exact_original_quote_checked", "claims_citations_checked",
                           "auxiliary_semantic_review" if state["draft"].claims else "no_policy_claims"]}

    @staticmethod
    def _after_validation(state: AgentState) -> str:
        return "repair" if state["validation_errors"] and state["repair_count"] < 1 and state["evidence"] else "finalize"

    def _repair(self, state: AgentState) -> dict:
        draft = self._model_call(state, "repair", lambda: self.provider.generate(
            state["request"].question, state["analysis"], state["evidence"],
            repair_errors=state["validation_errors"][:8]))
        if not isinstance(draft, AnswerDraft):
            draft = AnswerDraft.model_validate(draft)
        return {"draft": draft, "repair_count": 1, "trace": [*state["trace"], "repair"],
                "checks": [*state["checks"], "single_repair_attempted"]}

    def _finalize(self, state: AgentState) -> dict:
        draft = state["draft"]
        errors = state["validation_errors"]
        evidence = state["evidence"]
        has_summary = any(clause.metadata.content_kind == "summary" for clause in evidence)
        if errors:
            draft = AnswerDraft(status="review_required",
                                answer="答案未通过引用或支持性校验，已停止发布制度结论。请查阅下方已过滤的资料并提交人工复核。")
        elif draft.claims:
            intro = "以下为相关资料，请结合业务事实核对适用条件。" if has_summary else (
                "以下为相关制度原文，请结合业务事实核对适用条件。")
            if draft.status == "clarification_needed":
                intro = "当前事实不足以认定操作符合制度，请结合以下资料补充业务事实。" if has_summary else (
                    "当前事实不足以认定操作符合制度，请结合以下原文补充业务事实。")
            quoted = [f"{claim.text} [{', '.join(claim.evidence_ids)}]" for claim in draft.claims]
            draft = draft.model_copy(update={
                "answer": intro + "\n\n" + "\n\n".join(quoted),
                "rationale_summary": "制度主张逐字摘录本次有效条款；模型支持性评审仅作为辅助检查。",
                "applicable_conditions": [claim.text for claim in draft.claims],
                # Free-form generated questions are another possible route for
                # unsupported policy assertions. Preserve only a fixed factual
                # clarification prompt; original conditions remain visible.
                "missing_information": ["请说明实际业务是否满足以上条款列出的条件，并补充尚未核实的事实。"]
                if draft.status == "clarification_needed" else [],
            })
        elif "analysis" in state and draft.status == "clarification_needed":
            missing = state["analysis"].missing_information if state["analysis"].needs_clarification else [
                "请补充具体业务事实，并说明哪些制度条件尚未核实。"]
            draft = draft.model_copy(update={"answer": "请补充以下业务事实后再查询。", "rationale_summary": "",
                                             "missing_information": missing, "applicable_conditions": []})
        elif draft.status in ("insufficient_evidence", "review_required") and "analysis" in state:
            draft = AnswerDraft(status=draft.status, answer="有效证据不足或存在待复核事项，当前无法给出制度结论。")
        if has_summary:
            draft = draft.model_copy(update={
                "answer": "【所引资料包含整理摘要（非官方原文），请通过来源链接核对官方资料。】\n" + draft.answer,
                "rationale_summary": "所引资料包含整理摘要（非官方原文），引用校验仅核对知识库资料；"
                "官方规则须通过来源链接核对，模型支持性评审仅作为辅助检查。",
            })
        if evidence and all(clause.metadata.document_type == "teaching" for clause in evidence):
            draft = draft.model_copy(update={"answer": "【教学示例，仅用于验证系统流程，不作为真实银行制度依据】\n" + draft.answer})
        response = QueryResponse(
            **draft.model_dump(),
            evidence=[EvidenceView(evidence_id=clause.evidence_id, title=clause.metadata.title,
                                   version=clause.metadata.version, article=clause.article,
                                   effective_date=clause.metadata.effective_date, text=clause.text,
                                   document_type=clause.metadata.document_type, page=clause.page,
                                   content_kind=clause.metadata.content_kind,
                                   source_effective_date=clause.metadata.source_effective_date,
                                   source_url=clause.metadata.source_url) for clause in evidence],
            snapshot_id=state["snapshot_id"], session_id=state["session_id"], trace_id=state["trace_id"],
            checks=list(dict.fromkeys([*state["checks"], "review_required" if errors else "validation_passed"])),
            demo_mode=self.settings.demo_mode,
        )
        state["trace"].append("finalize")
        self._audit(state, response)
        return {"response": response, "trace": state["trace"]}

    def _audit(self, state: AgentState, response: QueryResponse) -> None:
        record = {
            "trace_id": state["trace_id"], "user_id": state["user"].user_id,
            "role": state["user"].role, "organization": state["user"].organization,
            "snapshot_id": response.snapshot_id, "status": response.status,
            "business_date": state["request"].business_date.isoformat(),
            "evidence_ids": [clause.evidence_id for clause in state["evidence"]],
            "versions": sorted({clause.metadata.policy_id + "@" + clause.metadata.version
                                for clause in state["evidence"]}),
            "rule_checks": response.checks, "nodes": list(state["trace"]),
            "supplement_rounds": state["supplement_rounds"], "subquestions_used": state["subquestions_used"],
            "repair_count": state["repair_count"], "model_id": self.provider.model_id,
            "model_calls": state["model_steps"], "usage": self.provider.usage_summary(),
            "retrieval_queries": state["retrieval_queries"],
            "duration_ms": round((time.perf_counter() - state["started_at"]) * 1000, 2),
        }
        self.store.audit(record)
