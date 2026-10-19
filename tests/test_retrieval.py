from datetime import date

import pytest

from bank_agent.config import Settings
from bank_agent.knowledge import KnowledgeStore
from bank_agent.retrieval import HybridRetriever
from bank_agent.schemas import DocumentMetadata, QueryRequest, UserContext


class TinyEmbeddings:
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, float("复核" in text), float("审批" in text)]


class RecordingReranker:
    def __init__(self):
        self.seen = []

    def rerank(self, query, hits):
        self.seen = hits
        return hits


def add(store, tmp_path, policy_id, text, **kwargs):
    path = tmp_path / f"{policy_id}.txt"
    path.write_text(text, encoding="utf-8")
    effective_date = kwargs.pop("effective_date", date(2025, 1, 1))
    store.ingest(path, DocumentMetadata(policy_id=policy_id, title=policy_id, version="1",
                                       effective_date=effective_date, reviewed=True,
                                       reviewed_by="tester", **kwargs))


def test_both_channels_only_see_allowed_candidates(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    store = KnowledgeStore(settings)
    store.register_index_builder(lambda snapshot_id: None)
    add(store, tmp_path, "allowed", "第一条 普通业务必须复核。")
    add(store, tmp_path, "hidden", "第一条 秘密复核审批，超级复核审批。", access_level="restricted")
    add(store, tmp_path, "future", "第一条 复核审批。", effective_date=date(2030, 1, 1))
    snapshot = store.publish()
    reranker = RecordingReranker()
    retriever = HybridRetriever(settings, store, TinyEmbeddings(), reranker)
    allowed_id = next(c.evidence_id for c in store.clauses() if c.metadata.policy_id == "allowed")

    class GuardedCollection:
        def query(self, **kwargs):
            assert kwargs["where"] == {"evidence_id": {"$in": [allowed_id]}}
            assert kwargs["n_results"] == 1
            return {"ids": [[allowed_id]]}

    monkeypatch.setattr(retriever, "_collection", lambda selected: GuardedCollection())
    import rank_bm25
    actual_bm25 = rank_bm25.BM25Okapi

    def guarded_bm25(corpus):
        assert len(corpus) == 1
        assert "秘密" not in "".join(corpus[0])
        return actual_bm25(corpus)

    monkeypatch.setattr(rank_bm25, "BM25Okapi", guarded_bm25)
    hits = retriever.search("复核", UserContext(user_id="u"),
                            QueryRequest(question="复核", business_date=date(2026, 1, 1)), snapshot)
    assert [h.clause.evidence_id for h in hits] == [allowed_id]
    assert hits[0].channels == ["vector_demo", "bm25"]
    assert reranker.seen == hits


def test_real_chroma_snapshot_and_embedding_identity(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    store = KnowledgeStore(settings)
    reranker = RecordingReranker()
    retriever = HybridRetriever(settings, store, TinyEmbeddings(), reranker)
    add(store, tmp_path, "public", "第一条 必须复核。")
    add(store, tmp_path, "restricted", "第一条 内部审批细节。", access_level="restricted")
    first = store.publish()
    request = QueryRequest(question="复核", business_date=date(2026, 1, 1))
    employee = UserContext(user_id="u")
    assert {h.clause.metadata.policy_id for h in retriever.search("复核", employee, request)} == {"public"}
    retriever.build_snapshot(first)  # Idempotent build.
    add(store, tmp_path, "extra", "第一条 新制度必须审批。")
    second = store.publish()
    assert retriever._collection_name(first) != retriever._collection_name(second)
    assert retriever._collection(first).count() == 2
    assert retriever._collection(second).count() == 3
    production = HybridRetriever(settings.model_copy(update={"demo_mode": False}), store,
                                 TinyEmbeddings(), reranker)
    with pytest.raises(RuntimeError, match="embedding 模型/模式"):
        production.search("复核", employee, request, first)


def test_vector_violation_stops_before_reranker(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    store = KnowledgeStore(settings)
    store.register_index_builder(lambda snapshot_id: None)
    add(store, tmp_path, "allowed", "第一条 普通业务必须复核。")
    store.publish()
    reranker = RecordingReranker()
    retriever = HybridRetriever(settings, store, TinyEmbeddings(), reranker)

    class BrokenCollection:
        def query(self, **kwargs):
            return {"ids": [["unauthorized-id"]]}

    monkeypatch.setattr(retriever, "_collection", lambda selected: BrokenCollection())
    with pytest.raises(RuntimeError, match="预过滤"):
        retriever.search("复核", UserContext(user_id="u"), QueryRequest(question="复核"))
    assert reranker.seen == []


def test_reranker_cannot_rewrite_authoritative_clause(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    store = KnowledgeStore(settings)
    store.register_index_builder(lambda snapshot_id: None)
    add(store, tmp_path, "allowed", "第一条 必须双人复核。")
    store.publish()
    original = store.clauses()[0]

    class MutatingReranker:
        def rerank(self, query, hits):
            hits[0].clause.text = "伪造：不需要复核。"
            hits[0].score = 0.9
            return [hits[0], hits[0]]

    class Collection:
        def query(self, **kwargs):
            return {"ids": [[original.evidence_id]]}

    retriever = HybridRetriever(settings, store, TinyEmbeddings(), MutatingReranker())
    monkeypatch.setattr(retriever, "_collection", lambda selected: Collection())
    hits = retriever.search("复核", UserContext(user_id="u"), QueryRequest(question="复核"))
    assert len(hits) == 1
    assert hits[0].clause.text == original.text
    assert hits[0].score == 0.9
