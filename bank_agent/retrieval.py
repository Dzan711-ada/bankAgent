"""Two prefiltered retrieval channels, reciprocal-rank fusion and reranking."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any

from .config import Settings
from .knowledge import KnowledgeStore
from .schemas import QueryRequest, SearchHit, UserContext


def tokenize(text: str) -> list[str]:
    import jieba

    jieba.setLogLevel(logging.WARNING)
    return [word.lower() for word in jieba.cut_for_search(text) if word.strip() and any(c.isalnum() for c in word)]


class HybridRetriever:
    def __init__(self, settings: Settings, store: KnowledgeStore, embeddings: Any, reranker: Any):
        self.settings = settings
        self.store = store
        self.embeddings = embeddings
        self.reranker = reranker
        self._client: Any = None
        identity = {
            "demo_mode": settings.demo_mode,
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "revision": settings.embedding_revision,
            "implementation": f"{type(embeddings).__module__}.{type(embeddings).__qualname__}",
            "base_url": settings.embedding_base_url,
            "custom_fingerprint": str(getattr(embeddings, "fingerprint", "")),
            "dimensions": getattr(embeddings, "dimensions", None),
        }
        self.embedding_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        store.register_index_builder(self.build_snapshot)

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=str(self.settings.data_dir / "chroma"),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @staticmethod
    def _collection_name(snapshot_id: str) -> str:
        # Never use a caller-controlled snapshot ID as a filesystem path/name.
        return "policy_" + hashlib.sha256(snapshot_id.encode()).hexdigest()[:40]

    def _collection(self, snapshot_id: str):
        try:
            collection = self.client.get_collection(name=self._collection_name(snapshot_id), embedding_function=None)
        except Exception as error:
            raise RuntimeError("快照向量索引不存在或无法读取，请重新发布知识快照") from error
        metadata = collection.metadata or {}
        if metadata.get("snapshot_id") != snapshot_id:
            raise RuntimeError("快照向量索引身份校验失败")
        if metadata.get("embedding_fingerprint") != self.embedding_fingerprint:
            raise RuntimeError("快照的 embedding 模型/模式与当前配置不同，请重新发布快照；不能混用演示和生产索引")
        if metadata.get("index_status") != "ready":
            raise RuntimeError("快照向量索引尚未构建完成")
        return collection

    def build_snapshot(self, snapshot_id: str) -> None:
        clauses = self.store.clauses(snapshot_id)
        if not clauses:
            raise ValueError("不能为无条款的快照构建索引")
        name = self._collection_name(snapshot_id)
        from chromadb.errors import NotFoundError

        try:
            existing = self.client.get_collection(name=name, embedding_function=None)
        except NotFoundError:
            existing = None
        if existing is not None:
            metadata = existing.metadata or {}
            if metadata.get("embedding_fingerprint") != self.embedding_fingerprint:
                raise RuntimeError("不能覆盖不同 embedding 配置的历史快照，请发布新快照")
            if metadata.get("index_status") == "ready" and existing.count() == len(clauses):
                return
            self.client.delete_collection(name=name)
        metadata = {"snapshot_id": snapshot_id, "embedding_fingerprint": self.embedding_fingerprint,
                    "index_status": "building", "demo_mode": self.settings.demo_mode}
        collection = self.client.create_collection(
            name=name,
            embedding_function=None,
            metadata=metadata,
            configuration={"hnsw": {"space": "cosine"}},
        )
        # API batches remain small enough for both local Chroma and remote
        # embedding services. Original text stays in the immutable SQLite store.
        for start in range(0, len(clauses), 64):
            batch = clauses[start:start + 64]
            vectors = self.embeddings.embed_documents([c.text for c in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("embedding 服务返回数量与条款数量不一致")
            collection.add(ids=[c.evidence_id for c in batch], embeddings=vectors,
                           metadatas=[{"evidence_id": c.evidence_id, "policy_id": c.metadata.policy_id,
                                       "version": c.metadata.version} for c in batch])
        if collection.count() != len(clauses):
            raise RuntimeError("快照索引条款数量校验失败")
        collection.modify(metadata={**metadata, "index_status": "ready"})

    def search(self, query: str, user: UserContext, request: QueryRequest,
               snapshot_id: str | None = None) -> list[SearchHit]:
        selected = self.store.active_snapshot() if snapshot_id is None else snapshot_id
        if not selected:
            return []
        # Freeze the exact same snapshot and authorized universe for BOTH
        # channels. Unauthorized clauses never enter BM25 or the vector query.
        allowed = self.store.allowed_clauses(user, request, selected)
        if not allowed:
            return []
        candidates = {c.evidence_id: c for c in allowed}
        collection = self._collection(selected)
        result = collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=min(self.settings.retrieval_k, len(allowed)),
            where={"evidence_id": {"$in": list(candidates)}},
            include=["distances"],
        )
        vector_ids = result["ids"][0]
        if any(evidence_id not in candidates for evidence_id in vector_ids):
            raise RuntimeError("向量索引违反预过滤约束，已中止检索")

        from rank_bm25 import BM25Okapi

        tokens = [tokenize(c.text) for c in allowed]
        query_tokens = tokenize(query)
        lexical_ids: list[str] = []
        if query_tokens and any(tokens):
            scores = BM25Okapi(tokens).get_scores(query_tokens)
            query_set = set(query_tokens)
            # BM25 may return negative scores for a tiny corpus. Token overlap,
            # rather than score > 0, decides whether there was a lexical match.
            matched = [(c.evidence_id, float(scores[index])) for index, c in enumerate(allowed)
                       if query_set.intersection(tokens[index])]
            lexical_ids = [identifier for identifier, _ in sorted(matched, key=lambda item: (-item[1], item[0]))
                           ][:self.settings.retrieval_k]

        fused: dict[str, SearchHit] = {}
        for channel, ranking in (("vector_demo" if self.settings.demo_mode else "vector", vector_ids),
                                 ("bm25", lexical_ids)):
            for rank, evidence_id in enumerate(ranking, start=1):
                hit = fused.setdefault(evidence_id, SearchHit(clause=candidates[evidence_id]))
                hit.score += 1.0 / (60 + rank)
                hit.channels.append(channel)
        hits = sorted(fused.values(), key=lambda hit: (-hit.score, hit.clause.evidence_id))[:self.settings.rerank_k]
        reranked = self.reranker.rerank(query, [hit.model_copy(deep=True) for hit in hits]) if hits else []
        permitted_ids = {hit.clause.evidence_id for hit in hits}
        if any(hit.clause.evidence_id not in permitted_ids for hit in reranked):
            raise RuntimeError("reranker 返回了候选集之外的条款")
        if any(not math.isfinite(hit.score) for hit in reranked):
            raise RuntimeError("reranker 返回了非有限分数")
        # A reranker assigns scores; it cannot change the authoritative source
        # text or attach a new clause under an existing evidence identifier.
        original_hits = {hit.clause.evidence_id: hit for hit in hits}
        seen: set[str] = set()
        safe_hits: list[SearchHit] = []
        for hit in reranked:
            identifier = hit.clause.evidence_id
            if identifier not in seen:
                safe_hits.append(original_hits[identifier].model_copy(update={"score": hit.score}))
                seen.add(identifier)
        return safe_hits[:self.settings.evidence_k]
