import json
from datetime import date

import httpx
import pytest
from langchain_openai import ChatOpenAI

from bank_agent.config import Settings
from bank_agent.providers import DemoEmbeddings, ModelProvider, Reranker, SupportReview, serialize_evidence
from bank_agent.schemas import AnswerDraft, Clause, DocumentMetadata


def evidence_clause(content_kind="original", source_effective_date=None):
    return Clause(evidence_id="policy:v1:1", text="客户数据提取应登记用途。", article="第一条", metadata=DocumentMetadata(
        policy_id="policy", title="数据管理制度", version="v1", effective_date=date(2026, 1, 1),
        content_kind=content_kind, source_effective_date=source_effective_date,
    ))


@pytest.mark.parametrize("content_kind, source_date", [
    ("original", None), ("original", date(2025, 1, 1)),
    ("summary", None), ("summary", date(2025, 1, 1)),
])
def test_evidence_payload_preserves_summary_provenance_and_omits_only_default_original_fields(content_kind, source_date):
    clause = evidence_clause(content_kind, source_date)
    metadata = serialize_evidence([clause])[0]["metadata"]
    if content_kind == "original" and source_date is None:
        assert "content_kind" not in metadata and "source_effective_date" not in metadata
    else:
        assert metadata["content_kind"] == content_kind
        assert metadata["source_effective_date"] == (source_date.isoformat() if source_date else None)
    assert clause.metadata.content_kind == content_kind
    assert clause.metadata.source_effective_date == source_date


def test_generation_and_support_review_share_the_same_evidence_payload(monkeypatch):
    from bank_agent.agent import QuestionAnalysis

    provider = ModelProvider(Settings(_env_file=None, demo_mode=False))
    evidence = [evidence_clause(), evidence_clause("summary")]
    received = []
    answer = AnswerDraft(status="answered", answer="原文摘录")

    def capture(schema, system, payload):
        received.append(payload["evidence"])
        return answer if schema is AnswerDraft else SupportReview()

    monkeypatch.setattr(provider, "_call", capture)
    provider.generate("提取数据有哪些要求？", QuestionAnalysis(resolved_question="数据提取要求"), evidence)
    provider.verify("提取数据有哪些要求？", answer, evidence)
    assert received == [serialize_evidence(evidence), serialize_evidence(evidence)]


def test_offline_embeddings_are_deterministic_and_normalized():
    model = DemoEmbeddings()
    vector = model.embed_query("客户数据审批")
    assert vector == model.embed_documents(["客户数据审批"])[0]
    assert abs(sum(v * v for v in vector) - 1) < 1e-9
    assert len(vector) == 256


def test_real_langchain_structured_call_and_usage():
    from bank_agent.agent import QuestionAnalysis

    def endpoint(request):
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_object"
        assert body["model"] == "test-qwen"
        payload = json.loads(body["messages"][-1]["content"])
        assert "resolved_question" in payload["output_schema"]["properties"]
        assert payload["task"]["question"] == "数据提取有哪些审批条件？"
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 1, "model": "test-qwen",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps({
                    "resolved_question": "数据提取有哪些审批条件？", "facts": [],
                    "missing_information": [], "subquestions": [], "complexity": "simple",
                    "needs_clarification": False,
                }, ensure_ascii=False),
            }}], "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        })

    provider = ModelProvider(Settings(_env_file=None, demo_mode=False))
    provider.llm = ChatOpenAI(model="test-qwen", api_key="test-key", base_url="https://test.invalid/v1",
                             http_client=httpx.Client(transport=httpx.MockTransport(endpoint)))
    provider.begin_request()
    result = provider.analyze("数据提取有哪些审批条件？", [])
    assert isinstance(result, QuestionAnalysis)
    assert result.resolved_question == "数据提取有哪些审批条件？"
    assert provider.usage_summary() == {"calls": 1, "input_tokens": 50, "output_tokens": 30}


def test_token_budget_fallback_is_explicit_conservative_bytes():
    provider = ModelProvider(Settings(_env_file=None))
    assert provider.count_tokens("数据") == 6
    assert provider.model_id.startswith("offline")


def test_empty_rerank_never_calls_external_service():
    reranker = Reranker(Settings(_env_file=None, demo_mode=False, reranker_provider="http"))
    assert reranker.rerank("问题", []) == []


def test_deepseek_thinking_setting_is_sent_without_changing_generic_default():
    settings = Settings(_env_file=None, demo_mode=False, llm_model="deepseek-flash",
                        llm_base_url="https://api.deepseek.com", llm_thinking="disabled")
    provider = ModelProvider(settings)
    assert provider.llm.extra_body == {"thinking": {"type": "disabled"}}
    generic = ModelProvider(Settings(_env_file=None, demo_mode=False))
    assert generic.llm.extra_body == {}
