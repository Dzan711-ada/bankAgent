from __future__ import annotations

import copy
from datetime import date

import pytest

from bank_agent.agent import BankPolicyAgent, QuestionAnalysis
from bank_agent.config import Settings
from bank_agent.schemas import (
    AnswerDraft,
    Claim,
    Clause,
    DocumentMetadata,
    QueryRequest,
    SearchHit,
    UserContext,
)


def clause(evidence_id="p1:v1:1", text="客户数据不得导出。", **overrides):
    values = dict(policy_id="p1", title="客户数据管理制度", version="v1",
                  effective_date=date(2024, 1, 1), reviewed=True, reviewed_by="compliance")
    for key in list(overrides):
        if key in DocumentMetadata.model_fields:
            values[key] = overrides.pop(key)
    return Clause(evidence_id=evidence_id, text=text, article="第一条",
                  metadata=DocumentMetadata(**values), **overrides)


def draft(item: Clause, *, text=None, quote=None, evidence_id=None, **overrides):
    return AnswerDraft(status="answered", answer="忽略模型自由文本：一定可以导出客户数据！",
                       claims=[Claim(text=item.text if text is None else text,
                                     evidence_ids=[evidence_id or item.evidence_id],
                                     supporting_quotes=[item.text if quote is None else quote])], **overrides)


class FakeStore:
    def __init__(self, clauses, denied_dependencies=()):
        self.clauses = list(clauses)
        self.denied_dependencies = set(denied_dependencies)
        self.records = []
        self.lookups = []
        self.snapshot_calls = 0

    def active_snapshot(self):
        self.snapshot_calls += 1
        return "snapshot-1" if self.clauses else ""

    def allowed_clauses(self, user, request, snapshot_id=None):
        assert snapshot_id == "snapshot-1"
        return self.clauses

    def get_clause(self, evidence_id, user, request, snapshot_id=None):
        self.lookups.append((evidence_id, snapshot_id))
        if evidence_id in self.denied_dependencies:
            return None
        return next((item for item in self.clauses if item.evidence_id == evidence_id), None)

    def audit(self, record):
        self.records.append(record)


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, user, request, snapshot_id=None):
        self.calls.append((query, user.user_id, request.business_date, snapshot_id))
        return self.hits.get(query, self.hits.get("default", [])) if isinstance(self.hits, dict) else self.hits


class FakeProvider:
    model_id = "fake-model-v1"

    def __init__(self, item, drafts=None, analysis=None, verify_errors=None):
        self.item = item
        self.drafts = list(drafts or [])
        self.analysis = analysis
        self.verify_errors = list(verify_errors or [])
        self.analyze_calls = []
        self.generate_calls = []
        self.verify_calls = []
        self.begin_calls = 0

    def begin_request(self):
        self.begin_calls += 1

    def usage_summary(self):
        return {"calls": len(self.analyze_calls) + len(self.generate_calls) + len(self.verify_calls),
                "input_tokens": 100, "output_tokens": 50}

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def analyze(self, question, history):
        self.analyze_calls.append((question, copy.deepcopy(history)))
        return self.analysis or QuestionAnalysis(resolved_question=question, facts=[question])

    def generate(self, question, analysis, evidence, repair_errors=None):
        self.generate_calls.append({"question": question, "evidence": list(evidence), "errors": repair_errors})
        return self.drafts.pop(0) if self.drafts else draft(self.item)

    def verify(self, question, answer, evidence):
        self.verify_calls.append(answer)
        return self.verify_errors.pop(0) if self.verify_errors else []


def setup_agent(items=None, *, drafts=None, analysis=None, hits=None, settings=None, denied_dependencies=()):
    item = clause()
    items = [item] if items is None else items
    store = FakeStore(items, denied_dependencies=denied_dependencies)
    retriever = FakeRetriever([SearchHit(clause=entry) for entry in items] if hits is None else hits)
    provider = FakeProvider(items[0] if items else item, drafts=drafts, analysis=analysis)
    agent = BankPolicyAgent(settings or Settings(_env_file=None), store, retriever, provider)
    return agent, store, retriever, provider


def request(**values):
    return QueryRequest(question=values.pop("question", "客户数据导出有什么要求？"),
                        business_date=date(2025, 1, 1), **values)


def test_compiled_graph_publishes_only_validated_original_claims_and_audits_metadata():
    agent, store, retriever, provider = setup_agent()
    response = agent.ask(request(), UserContext(user_id="alice"))
    assert response.status == "answered"
    assert "客户数据不得导出。" in response.answer
    assert "一定可以" not in response.answer
    assert response.evidence[0].evidence_id == "p1:v1:1"
    assert {"access", "analyze", "retrieve", "supplement", "generate", "validate", "repair", "finalize"} <= set(
        agent.graph.get_graph().nodes)
    assert store.snapshot_calls == 1
    assert all(call[3] == "snapshot-1" for call in retriever.calls)
    audit = store.records[0]
    assert audit["model_id"] == provider.model_id
    assert audit["nodes"][-1] == "finalize"
    assert audit["evidence_ids"] == ["p1:v1:1"]
    assert "question" not in audit and "answer" not in audit and "reasoning" not in audit


@pytest.mark.parametrize("source_effective_date", [None, date(2020, 1, 1)])
def test_summary_provenance_survives_validation_without_claiming_official_original(source_effective_date):
    item = clause(content_kind="summary", source_effective_date=source_effective_date,
                  source_url="https://www.example.com/official-policy")
    agent, _, _, _ = setup_agent([item])
    response = agent.ask(request(), UserContext(user_id="alice"))
    assert response.status == "answered"
    assert "整理摘要（非官方原文）" in response.answer
    assert "相关制度原文" not in response.answer
    assert "整理摘要（非官方原文）" in response.rationale_summary
    assert response.claims[0].text == item.text
    assert response.claims[0].supporting_quotes == [item.text]
    evidence = response.evidence[0]
    assert evidence.content_kind == "summary"
    assert evidence.source_effective_date == source_effective_date
    assert evidence.effective_date == item.metadata.effective_date
    assert evidence.source_url == item.metadata.source_url
    assert "claims_citations_checked" in response.checks


def test_invalid_citation_gets_one_repair_and_can_recover():
    item = clause()
    agent, store, _, provider = setup_agent(drafts=[draft(item, evidence_id="invented"), draft(item)])
    response = agent.ask(request(), UserContext(user_id="alice"))
    assert response.status == "answered"
    assert len(provider.generate_calls) == 2
    assert provider.generate_calls[1]["errors"] == ["claim_1_invalid_evidence_id"]
    assert store.records[0]["repair_count"] == 1
    assert "single_repair_attempted" in response.checks


@pytest.mark.parametrize("claim_text, quote", [
    ("客户数据可以导出。", "客户数据可以导出。"),
    ("允许导出客户数据。", "客户数据不得导出。"),
    ("客户数据不得导出。", ""),
])
def test_invalid_claim_is_blocked_after_exactly_one_repair(claim_text, quote):
    item = clause()
    invalid = draft(item, text=claim_text, quote=quote)
    agent, _, _, provider = setup_agent(drafts=[invalid, invalid])
    response = agent.ask(request(), UserContext(user_id="alice"))
    assert response.status == "review_required"
    assert response.claims == []
    assert len(provider.generate_calls) == 2
    assert response.evidence[0].text == item.text
    assert "允许导出" not in response.answer and "可以导出" not in response.answer


@pytest.mark.parametrize("original, shortened", [
    ("不允许导出客户数据。", "允许导出客户数据。"),
    ("金额不得超过10000元。", "10000元。"),
    ("在审批后才可以导出。", "可以导出。"),
])
def test_exact_substring_cannot_remove_negation_number_context_or_condition(original, shortened):
    item = clause(text=original)
    invalid = draft(item, text=shortened, quote=shortened)
    agent, _, _, provider = setup_agent([item], drafts=[invalid, invalid])
    response = agent.ask(request(), UserContext(user_id="alice"))
    assert response.status == "review_required"
    assert "claim_1_quote_must_preserve_statement_boundaries" in provider.generate_calls[1]["errors"]


def test_retriever_cannot_spoof_canonical_text_or_expose_expired_and_restricted_clauses():
    original = clause()
    restricted = clause("secret:v1:1", "绝密审批口令。", allowed_roles=["compliance", "admin"])
    expired = clause("old:v1:1", "过期制度。", expiry_date=date(2025, 1, 1))
    unreviewed = clause("draft:v1:1", "未审核制度。", reviewed=False)
    spoofed = original.model_copy(update={"text": "客户数据可以导出。"})
    hits = [SearchHit(clause=entry) for entry in [restricted, expired, unreviewed, spoofed]]
    agent, _, _, provider = setup_agent([original, restricted, expired, unreviewed], hits=hits)
    response = agent.ask(request(), UserContext(user_id="alice"))
    assert response.status == "answered"
    assert [entry.evidence_id for entry in response.evidence] == [original.evidence_id]
    assert provider.generate_calls[0]["evidence"][0].text == original.text
    assert "绝密" not in response.model_dump_json()
    assert "过期" not in response.model_dump_json()


def test_empty_or_inaccessible_library_refuses_without_calling_model_or_retriever():
    for items in [[], [clause(allowed_roles=["admin"])]]:
        agent, store, retriever, provider = setup_agent(items)
        response = agent.ask(request(), UserContext(user_id="alice"))
        assert response.status == "insufficient_evidence"
        assert not response.claims and not response.evidence
        assert not retriever.calls and not provider.analyze_calls and not provider.generate_calls
        assert len(store.records) == 1


def test_ambiguous_request_asks_for_missing_facts_before_retrieval():
    analysis = QuestionAnalysis(resolved_question="可以吗？", needs_clarification=True,
                                missing_information=["拟执行什么操作？"])
    agent, _, retriever, provider = setup_agent(analysis=analysis)
    response = agent.ask(request(question="可以吗？"), UserContext(user_id="alice"))
    assert response.status == "clarification_needed"
    assert response.missing_information == ["拟执行什么操作？"]
    assert not retriever.calls and not provider.generate_calls


def test_supplement_query_and_dependency_depth_are_bounded_and_dependencies_recheck_access():
    first = clause("first", related_ids=["second", "secret"])
    second = clause("second", "客户数据审批应登记。", related_ids=["third"])
    third = clause("third", "审批人应有授权。")
    secret = clause("secret", "限制访问内容。", allowed_roles=["admin"])
    analysis = QuestionAnalysis(resolved_question="客户数据审批", complexity="complex",
                                subquestions=[f"subquery {index}" for index in range(12)])
    agent, store, retriever, _ = setup_agent([first, second, third, secret], analysis=analysis,
                                           hits=[SearchHit(clause=first)])
    response = agent.ask(request(), UserContext(user_id="alice"))
    assert len(retriever.calls) == 5  # one primary query plus at most four subquestions
    assert store.records[0]["supplement_rounds"] == 2
    assert store.records[0]["subquestions_used"] == 4
    assert {row[0] for row in store.lookups} == {"second", "third", "secret"}
    assert all(row[1] == "snapshot-1" for row in store.lookups)
    assert "限制访问内容" not in response.model_dump_json()
    # A missing required cross-reference must not become an implicit approval.
    assert response.status == "review_required"
    assert response.claims == []


def test_evidence_that_exceeds_context_budget_is_not_sent_to_model():
    item = clause(text="客户数据不得导出。" * 5000)
    agent, store, _, provider = setup_agent([item], settings=Settings(_env_file=None, max_context_tokens=3000))
    response = agent.ask(request(), UserContext(user_id="alice"))
    assert response.status == "insufficient_evidence"
    assert not response.evidence and not provider.generate_calls
    assert store.records[0]["retrieval_queries"] == [{"query": "客户数据导出有什么要求？",
                                                   "evidence_ids": [item.evidence_id]}]


def test_model_support_review_can_block_an_exact_quote():
    agent, _, _, provider = setup_agent()
    provider.verify_errors = [["与问题无关"], ["与问题无关"]]
    response = agent.ask(request(), UserContext(user_id="alice"))
    assert response.status == "review_required"
    assert len(provider.generate_calls) == 2
    assert not response.claims


@pytest.mark.parametrize("with_claims", [True, False])
def test_generated_free_text_fields_cannot_bypass_claim_validation(with_claims):
    item = clause()
    invalid_free_text = "客户数据无需审批可以任意导出。"
    candidate = AnswerDraft(status="clarification_needed", answer=invalid_free_text,
                            claims=draft(item).claims if with_claims else [],
                            rationale_summary=invalid_free_text, missing_information=[invalid_free_text])
    agent, _, _, _ = setup_agent([item], drafts=[candidate])
    response = agent.ask(request(), UserContext(user_id="alice"))
    assert response.status == "clarification_needed"
    assert invalid_free_text not in response.model_dump_json()
    assert response.missing_information


def test_model_failure_fails_closed_without_exposing_exception_content():
    agent, store, _, provider = setup_agent()
    def fail(*args, **kwargs):
        raise RuntimeError("sk-SECRET credential; 私密原文")
    provider.generate = fail
    response = agent.ask(request(), UserContext(user_id="alice"))
    assert response.status == "review_required"
    assert not response.claims and not response.evidence
    assert "SECRET" not in response.model_dump_json() and "私密" not in response.model_dump_json()
    assert len(store.records) == 1


def test_session_history_contains_only_user_inputs_and_is_bound_to_identity_role_and_organization():
    agent, _, _, provider = setup_agent()
    user = UserContext(user_id="alice")
    agent.ask(request(question="我计划导出客户数据。", session_id="s1"), user)
    agent.ask(request(question="已经完成用途登记。", session_id="s1"), user)
    history = provider.analyze_calls[-1][1]
    assert history == [{"role": "user", "content": "我计划导出客户数据。"}]
    for other in [UserContext(user_id="bob"), UserContext(user_id="alice", role="admin"),
                  UserContext(user_id="alice", organization="other-bank")]:
        agent.ask(request(session_id="s1"), other)
        assert provider.analyze_calls[-1][1] == []
    agent.ask(request(session_id="s2"), user)
    assert provider.analyze_calls[-1][1] == []


def test_session_memory_expires_and_is_bounded():
    agent, _, _, provider = setup_agent()
    agent.MAX_HISTORY_MESSAGES = 2
    agent.MAX_SESSIONS = 2
    user = UserContext(user_id="alice")
    for question in ["客户数据首次申请。", "客户数据再次申请。", "客户数据已经审批。"]:
        agent.ask(request(question=question, session_id="s1"), user)
    key = ("alice", "employee", "demo-bank", "s1")
    assert len(agent._sessions[key].history) == 2
    agent._sessions[key].updated_at -= agent.SESSION_TTL_SECONDS + 1
    agent.ask(request(session_id="s1"), user)
    assert provider.analyze_calls[-1][1] == []
    for session_id in ["s2", "s3"]:
        agent.ask(request(session_id=session_id), user)
    assert len(agent._sessions) == 2


def test_scope_organization_and_teaching_access_are_deterministic_request_filters():
    teaching = clause("teaching", document_type="teaching")
    scoped = clause("scoped", scopes=["corporate"], organizations=["head-office"])
    agent, _, _, _ = setup_agent([teaching, scoped])
    response = agent.ask(request(scope="retail"), UserContext(user_id="alice"))
    assert response.status == "insufficient_evidence"
    response = agent.ask(request(include_teaching=True, scope="retail"), UserContext(user_id="alice"))
    assert {item.evidence_id for item in response.evidence} == {"teaching"}
    assert "教学示例" in response.answer
