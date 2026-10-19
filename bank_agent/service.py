from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import RLock

from bank_agent.agent import BankPolicyAgent
from bank_agent.config import Settings, get_settings
from bank_agent.knowledge import KnowledgeStore
from bank_agent.providers import ModelProvider, Reranker, create_embeddings
from bank_agent.retrieval import HybridRetriever
from bank_agent.schemas import DocumentMetadata, QueryRequest, QueryResponse, UserContext


class AppService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.store = KnowledgeStore(self.settings)
        self.retriever = HybridRetriever(
            self.settings, self.store, create_embeddings(self.settings), Reranker(self.settings)
        )
        self.provider = ModelProvider(self.settings)
        self.agent = BankPolicyAgent(self.settings, self.store, self.retriever, self.provider)
        self._publish_lock = RLock()

    def bootstrap_demo(self) -> str:
        if not self.settings.demo_mode:
            raise ValueError("教学样例仅可在 DEMO_MODE=true 时初始化")
        examples = Path(__file__).resolve().parent.parent / "examples"
        with self._publish_lock:
            for name in ["data-use-v1", "data-use-v2", "authorization"]:
                metadata = DocumentMetadata.model_validate_json(
                    (examples / "metadata" / f"{name}.json").read_text(encoding="utf-8")
                )
                self.ingest(examples / "policies" / f"{name}.txt", metadata)
            return self.publish()

    def ingest(self, path: Path, metadata: DocumentMetadata):
        with self._publish_lock:
            return self.store.ingest(path, metadata)

    def publish(self) -> str:
        with self._publish_lock:
            return self.store.publish()

    def activate(self, snapshot_id: str) -> None:
        with self._publish_lock:
            self.retriever.build_snapshot(snapshot_id)
            self.store.activate(snapshot_id)

    def query(self, request: QueryRequest, user: UserContext) -> QueryResponse:
        return self.agent.ask(request, user)


@lru_cache
def get_service() -> AppService:
    return AppService()
