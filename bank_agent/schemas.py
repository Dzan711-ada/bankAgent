from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator


def business_today() -> date:
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


class UserContext(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    role: Literal["employee", "compliance", "admin"] = "employee"
    organization: str = "demo-bank"


class DocumentMetadata(BaseModel):
    policy_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    title: str = Field(min_length=1)
    version: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    effective_date: date
    expiry_date: date | None = None
    issuer: str = ""
    document_number: str = ""
    source_url: str | None = None
    access_level: Literal["public", "internal", "restricted"] = "public"
    allowed_roles: list[str] = Field(default_factory=lambda: ["employee", "compliance", "admin"])
    organizations: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    document_type: Literal["public", "teaching", "internal"] = "public"
    content_kind: Literal["original", "summary"] = "original"
    source_effective_date: date | None = None
    reviewed: bool = False
    reviewed_by: str = ""

    @model_validator(mode="after")
    def validate_dates(self):
        if self.expiry_date and self.expiry_date <= self.effective_date:
            raise ValueError("expiry_date 必须晚于 effective_date；废止日期当天不再有效")
        if self.reviewed and not self.reviewed_by.strip():
            raise ValueError("已审核文件必须登记 reviewed_by")
        return self


class Clause(BaseModel):
    evidence_id: str
    text: str
    article: str
    metadata: DocumentMetadata
    page: int | None = None
    related_ids: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    clause: Clause
    score: float = 0.0
    channels: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    business_date: date = Field(default_factory=business_today)
    scope: str | None = None
    include_teaching: bool = False
    session_id: str | None = Field(default=None, max_length=128)


class Claim(BaseModel):
    text: str = Field(description="必须等于 supporting_quotes 的原文；多段原文只以换行拼接，不得概括或改写。")
    evidence_ids: list[str] = Field(default_factory=list)
    # Exact supporting excerpts are checked against the retrieved originals.
    supporting_quotes: list[str] = Field(default_factory=list, description="按 evidence_ids 顺序完整复制对应 evidence.text。")


class AnswerDraft(BaseModel):
    status: Literal["answered", "clarification_needed", "insufficient_evidence", "review_required"]
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    applicable_conditions: list[str] = Field(
        default_factory=list, description="可返回 []。每项必须完整逐字复制某条 evidence.text，不得概括、改写、加前缀或截断。"
    )
    rationale_summary: str = ""
    missing_information: list[str] = Field(default_factory=list)


class EvidenceView(BaseModel):
    evidence_id: str
    title: str
    version: str
    article: str
    effective_date: date
    text: str
    document_type: str
    content_kind: Literal["original", "summary"] = "original"
    source_effective_date: date | None = None
    page: int | None = None
    source_url: str | None = None


class QueryResponse(AnswerDraft):
    evidence: list[EvidenceView] = Field(default_factory=list)
    snapshot_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    checks: list[str] = Field(default_factory=list)
    demo_mode: bool = False
