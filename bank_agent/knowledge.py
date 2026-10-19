"""Immutable policy versions, clause parsing, access control and atomic snapshots.

Source files are data, never instructions. Only human-reviewed versions can enter
the knowledge base. Publishing freezes clause contents before building an index;
the active pointer changes only after that index has been successfully built.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .schemas import Clause, DocumentMetadata, QueryRequest, UserContext


@dataclass(frozen=True)
class _Block:
    text: str
    page: int | None = None
    table: bool = False


_ARTICLE = re.compile(r"^\s*(第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?)")
_REFERENCE = re.compile(r"第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?")
_CHAPTER = re.compile(r"^\s*(第[一二三四五六七八九十百千万零〇两\d]+[章节])")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _table_text(rows: list[list[str | None]]) -> str:
    # Repeat no headers and drop no columns: monetary thresholds and exceptions
    # must remain together. Pipes make cell boundaries explicit to the reader.
    return "\n".join(" | ".join((cell or "").replace("\n", "；").strip() for cell in row) for row in rows)


def _read_blocks(path: Path) -> list[_Block]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        # UTF-8 BOM and the common Chinese Windows encoding are both supported.
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("gb18030")
        blocks: list[_Block] = []
        table_lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            # Markdown pipe tables retain headers, delimiters and all rows in
            # one unit; long tables must not lose a threshold's column labels.
            if suffix == ".md" and "|" in line:
                table_lines.append(line)
                continue
            if table_lines:
                blocks.append(_Block("\n".join(table_lines), table=True))
                table_lines = []
            if line:
                blocks.append(_Block(line))
        if table_lines:
            blocks.append(_Block("\n".join(table_lines), table=True))
        return blocks
    if suffix == ".docx":
        from docx import Document
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = Document(path)
        blocks: list[_Block] = []
        # document.paragraphs alone loses the ordering of paragraphs and tables.
        for element in document.element.body.iterchildren():
            if element.tag == qn("w:p"):
                text = Paragraph(element, document).text.strip()
                if text:
                    blocks.append(_Block(text))
            elif element.tag == qn("w:tbl"):
                table = Table(element, document)
                text = _table_text([[cell.text for cell in row.cells] for row in table.rows])
                if text.strip(" |\n"):
                    blocks.append(_Block(text, table=True))
        return blocks
    if suffix == ".pdf":
        import pdfplumber

        blocks = []
        with pdfplumber.open(path) as document:
            for page_number, page in enumerate(document.pages, start=1):
                if not page.chars and page.images:
                    raise ValueError(f"PDF 第{page_number}页为无文字图像：OCR 尚未实现，请先 OCR 并人工复核")
                tables = page.find_tables()
                ordered: list[tuple[float, _Block]] = []
                for line in page.extract_text_lines():
                    center_y = (line["top"] + line["bottom"]) / 2
                    center_x = (line["x0"] + line["x1"]) / 2
                    if any(t.bbox[0] <= center_x <= t.bbox[2] and t.bbox[1] <= center_y <= t.bbox[3]
                           for t in tables):
                        continue
                    if line["text"].strip():
                        ordered.append((line["top"], _Block(line["text"].strip(), page_number)))
                for table in tables:
                    text = _table_text(table.extract())
                    if text.strip(" |\n"):
                        ordered.append((table.bbox[1], _Block(text, page_number, True)))
                blocks.extend(block for _, block in sorted(ordered, key=lambda item: item[0]))
        if not blocks:
            raise ValueError("PDF 未提取到文字：扫描件 OCR 尚未实现，请先进行 OCR 并人工复核后导入")
        return blocks
    raise ValueError("仅支持 .pdf、.docx、.txt、.md；旧版 .doc 请先转换为 .docx")


def _split_clauses(blocks: list[_Block], metadata: DocumentMetadata, settings: Settings) -> list[Clause]:
    units: list[tuple[str, str, int | None, bool]] = []
    current: list[str] = []
    article = "正文"
    chapter = ""
    page: int | None = None
    table_number = 0

    def flush() -> None:
        nonlocal current, page
        if current:
            units.append((article, "\n".join(current), page, False))
        current, page = [], None

    for block in blocks:
        if block.table:
            flush()
            table_number += 1
            units.append((f"{article}·表{table_number}", block.text, block.page, True))
            continue
        heading = block.text.lstrip("#>*- \t")
        chapter_match = _CHAPTER.match(heading)
        match = _ARTICLE.match(heading)
        if chapter_match:
            flush()
            chapter = block.text
            article = chapter_match.group(1)
        elif match:
            flush()
            article = match.group(1)
        if not current:
            page = block.page
            # A chapter title supplies local context without borrowing another
            # article's obligations. The source line itself is retained too.
            if chapter and match:
                current.append(chapter)
        current.append(block.text)
    flush()

    clauses: list[Clause] = []
    article_groups: dict[str, list[Clause]] = {}
    for unit_number, (article, text, page, is_table) in enumerate(units):
        # Articles are the semantic boundary. Very long articles get overlapping
        # windows; every sibling has related_ids so the agent can supplement it.
        chunks = [text]
        if not is_table and len(text) > settings.chunk_size:
            step = settings.chunk_size - settings.chunk_overlap
            chunks = [text[start:start + settings.chunk_size] for start in range(0, len(text), step)]
        siblings: list[Clause] = []
        for chunk_number, chunk in enumerate(chunks):
            digest = hashlib.sha256(
                f"{metadata.policy_id}|{metadata.version}|{unit_number}|{chunk_number}|{chunk}".encode()
            ).hexdigest()[:20]
            siblings.append(Clause(
                evidence_id=f"{metadata.policy_id}:{metadata.version}:{digest}",
                text=chunk,
                article=article if len(chunks) == 1 else f"{article}（片段{chunk_number + 1}/{len(chunks)}）",
                page=page,
                metadata=metadata,
            ))
        for clause in siblings:
            clause.related_ids = [s.evidence_id for s in siblings if s.evidence_id != clause.evidence_id]
        clauses.extend(siblings)
        article_groups.setdefault(article.split("·表", 1)[0], []).extend(siblings)
    for clause in clauses:
        article_name = clause.article.split("（片段", 1)[0].split("·表", 1)[0]
        related = {c.evidence_id for c in article_groups.get(article_name, []) if c.evidence_id != clause.evidence_id}
        # An explicit cross-reference or an attached table supplies supplemental
        # context, without merging unrelated versions or permissions.
        for reference in _REFERENCE.finditer(clause.text):
            related.update(c.evidence_id for c in article_groups.get(reference.group(0), [])
                           if c.evidence_id != clause.evidence_id)
        clause.related_ids = sorted(related)
    return clauses


class KnowledgeStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = settings.data_dir / "knowledge.sqlite3"
        self._index_builder: Callable[[str], None] | None = None
        self._publish_lock = threading.RLock()
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    policy_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    clauses_json TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    PRIMARY KEY (policy_id, version)
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    clause_count INTEGER NOT NULL,
                    manifest_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshot_documents (
                    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
                    policy_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, policy_id, version),
                    FOREIGN KEY (policy_id, version) REFERENCES documents(policy_id, version)
                );
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connection(self):
        # No shared connection crosses FastAPI worker threads.
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def register_index_builder(self, builder: Callable[[str], None]) -> None:
        self._index_builder = builder

    def ingest(self, path: Path, metadata: DocumentMetadata) -> list[Clause]:
        if not metadata.reviewed or not metadata.reviewed_by.strip():
            raise ValueError("导入制度须先经业务人员人工审核，并登记 reviewed=true 与 reviewed_by")
        blocks = _read_blocks(Path(path))
        if not blocks:
            raise ValueError("文档没有可导入的正文或表格")
        metadata_values = metadata.model_dump(mode="json")
        metadata_json = _json(metadata_values)
        digest_metadata = dict(metadata_values)
        # New provenance defaults must not change a pre-existing original's
        # immutable digest. Explicit summaries and source dates remain hashed.
        if digest_metadata["content_kind"] == "original":
            digest_metadata.pop("content_kind")
        if digest_metadata["source_effective_date"] is None:
            digest_metadata.pop("source_effective_date")
        digest = hashlib.sha256(_json({
            "blocks": [{"text": b.text, "page": b.page, "table": b.table} for b in blocks],
            "metadata": digest_metadata,
        }).encode()).hexdigest()
        clauses = _split_clauses(blocks, metadata, self.settings)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT digest, clauses_json FROM documents WHERE policy_id=? AND version=?",
                (metadata.policy_id, metadata.version),
            ).fetchone()
            if existing:
                if existing["digest"] != digest:
                    raise ValueError("同一 policy_id + version 的内容或元数据已登记，不能覆盖；请使用新版本号")
                return [Clause.model_validate(item) for item in json.loads(existing["clauses_json"])]
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (metadata.policy_id, metadata.version, digest, metadata_json,
                 _json([c.model_dump(mode="json") for c in clauses]), _now()),
            )
        return clauses

    def publish(self) -> str:
        if self._index_builder is None:
            raise RuntimeError("发布前必须初始化 HybridRetriever 并注册索引构建器")
        with self._publish_lock:
            snapshot_id = f"snap_{uuid.uuid4().hex}"
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                documents = connection.execute(
                    "SELECT * FROM documents ORDER BY policy_id, version"
                ).fetchall()
                if not documents:
                    raise ValueError("知识库为空，不能发布快照")
                if any(not json.loads(d["metadata_json"])["reviewed"] for d in documents):
                    raise ValueError("存在未经人工审核的制度，不能发布")
                effective_versions: set[tuple[str, str]] = set()
                for document in documents:
                    key = (document["policy_id"], json.loads(document["metadata_json"])["effective_date"])
                    if key in effective_versions:
                        raise ValueError("同一制度存在生效日期相同的多个版本，无法确定版本优先级；请先修正版本登记")
                    effective_versions.add(key)
                clause_count = sum(len(json.loads(d["clauses_json"])) for d in documents)
                manifest = [{"policy_id": d["policy_id"], "version": d["version"], "digest": d["digest"]}
                            for d in documents]
                connection.execute("INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
                                   (snapshot_id, _now(), "building", clause_count, _json(manifest)))
                connection.executemany("INSERT INTO snapshot_documents VALUES (?, ?, ?)",
                                       [(snapshot_id, d["policy_id"], d["version"]) for d in documents])
            try:
                self._index_builder(snapshot_id)
            except Exception:
                with self._connection() as connection:
                    connection.execute("UPDATE snapshots SET status='failed' WHERE snapshot_id=?", (snapshot_id,))
                raise
            # Both writes commit atomically; readers see the previous or new
            # complete index. Old snapshots and all original versions survive.
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("UPDATE snapshots SET status='ready' WHERE snapshot_id=?", (snapshot_id,))
                connection.execute("INSERT INTO state VALUES ('active_snapshot', ?) "
                                   "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (snapshot_id,))
            return snapshot_id

    def active_snapshot(self) -> str:
        with self._connection() as connection:
            row = connection.execute("SELECT value FROM state WHERE key='active_snapshot'").fetchone()
        return row["value"] if row else ""

    def clauses(self, snapshot_id: str | None = None) -> list[Clause]:
        selected = self.active_snapshot() if snapshot_id is None else snapshot_id
        if not selected:
            return []
        with self._connection() as connection:
            if not connection.execute("SELECT 1 FROM snapshots WHERE snapshot_id=?", (selected,)).fetchone():
                raise ValueError("快照不存在")
            rows = connection.execute(
                "SELECT d.clauses_json FROM documents d JOIN snapshot_documents s "
                "ON d.policy_id=s.policy_id AND d.version=s.version WHERE s.snapshot_id=? "
                "ORDER BY d.policy_id, d.version", (selected,),
            ).fetchall()
        return [Clause.model_validate(item) for row in rows for item in json.loads(row["clauses_json"])]

    def allowed_clauses(self, user: UserContext, request: QueryRequest,
                        snapshot_id: str | None = None) -> list[Clause]:
        selected = self.active_snapshot() if snapshot_id is None else snapshot_id
        if not selected:
            return []
        with self._connection() as connection:
            row = connection.execute("SELECT status FROM snapshots WHERE snapshot_id=?", (selected,)).fetchone()
        # The builder alone uses clauses() to read a staging snapshot. End-user
        # evidence access cannot read a failed or unfinished publication.
        if not row or row["status"] != "ready":
            return []
        clauses = self.clauses(selected)
        latest: dict[str, date] = {}
        for clause in clauses:
            metadata = clause.metadata
            if metadata.effective_date <= request.business_date:
                latest[metadata.policy_id] = max(latest.get(metadata.policy_id, metadata.effective_date),
                                                 metadata.effective_date)
        allowed: list[Clause] = []
        for clause in clauses:
            metadata = clause.metadata
            if metadata.effective_date != latest.get(metadata.policy_id):
                continue
            # Version selection precedes ACL: inability to read the current
            # version must never reveal a superseded version as today's rule.
            if metadata.expiry_date and request.business_date >= metadata.expiry_date:
                continue
            if user.role not in metadata.allowed_roles:
                continue
            if metadata.access_level == "restricted" and user.role not in {"compliance", "admin"}:
                continue
            if metadata.organizations and user.organization not in metadata.organizations:
                continue
            if request.scope and metadata.scopes and request.scope not in metadata.scopes:
                continue
            if metadata.document_type == "teaching" and not request.include_teaching:
                continue
            if not metadata.reviewed:
                continue
            allowed.append(clause)
        return allowed

    def get_clause(self, evidence_id: str, user: UserContext, request: QueryRequest,
                   snapshot_id: str | None = None) -> Clause | None:
        return next((c for c in self.allowed_clauses(user, request, snapshot_id) if c.evidence_id == evidence_id), None)

    def snapshots(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM snapshots ORDER BY created_at DESC").fetchall()
        active = self.active_snapshot()
        return [{"snapshot_id": row["snapshot_id"], "status": row["status"],
                 "created_at": row["created_at"], "clause_count": row["clause_count"],
                 "active": row["snapshot_id"] == active, "manifest": json.loads(row["manifest_json"])}
                for row in rows]

    def activate(self, snapshot_id: str) -> None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
            if not row or row["status"] != "ready":
                raise ValueError("只能激活已成功建索引的 ready 快照")
            connection.execute("INSERT INTO state VALUES ('active_snapshot', ?) "
                               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (snapshot_id,))

    def audit(self, record: dict) -> None:
        # An allowlist prevents accidentally persisting prompts, complete evidence
        # or provider settings. Even approved string fields are redacted again.
        allowed_keys = {"trace_id", "session_id", "snapshot_id", "status", "question", "user_id", "role",
                        "organization", "evidence_ids", "checks", "duration_ms", "demo_mode",
                        "llm_model", "embedding_model", "reranker_model", "error_type", "business_date",
                        "versions", "rule_checks", "nodes", "supplement_rounds", "subquestions_used",
                        "repair_count", "model_id", "model_calls", "usage", "retrieval_queries"}
        nested_keys = {
            "model_calls": {"step", "succeeded", "duration_ms", "usage", "model_id"},
            "usage": {"calls", "input_tokens", "output_tokens", "total_tokens", "cached_tokens"},
            "retrieval_queries": {"query", "evidence_ids"},
        }

        def redact(value: str) -> str:
            value = re.sub(r"(?i)(api[_-]?key|token|password|密钥|密码)\s*[:=：]\s*\S+", r"\1=[REDACTED]", value)
            value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED_KEY]", value)
            value = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[REDACTED_ID]", value)
            value = re.sub(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)", "[REDACTED_PHONE]", value)
            value = re.sub(r"(?<!\d)\d{12,19}(?!\d)", "[REDACTED_NUMBER]", value)
            value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", value)
            return value[:1000]

        def sanitize(value: Any, field: str, depth: int = 0) -> Any:
            if depth > 4:
                return None
            if isinstance(value, str):
                return redact(value)
            if isinstance(value, (int, float, bool)) or value is None:
                return value
            if isinstance(value, list):
                return [sanitize(item, field, depth + 1) for item in value[:100]
                        if not isinstance(item, dict) or field in nested_keys]
            if isinstance(value, dict) and field in nested_keys:
                return {key: sanitize(item, key, depth + 1) for key, item in value.items()
                        if key in nested_keys[field]}
            return None

        payload: dict[str, Any] = {}
        for key, value in record.items():
            if key not in allowed_keys:
                continue
            if key == "user_id":
                payload[key] = "sha256:" + hashlib.sha256(str(value).encode()).hexdigest()[:24]
            else:
                sanitized = sanitize(value, key)
                if sanitized is not None or value is None:
                    payload[key] = sanitized
        with self._connection() as connection:
            connection.execute("INSERT INTO audit_log(created_at, payload_json) VALUES (?, ?)", (_now(), _json(payload)))
