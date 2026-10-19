from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from bank_agent.api import create_app
from bank_agent.config import Settings
from bank_agent.schemas import Clause, DocumentMetadata, QueryResponse


class FakeStore:
    def __init__(self):
        self.clause = Clause(
            evidence_id="policy-1-v1-article1",
            article="第一条",
            text="仅合规部门可以查阅的制度证据。",
            metadata=DocumentMetadata(
                policy_id="policy-1", title="受限制度", version="1",
                effective_date=date(2026, 1, 1), expiry_date=date(2027, 1, 1),
                allowed_roles=["compliance", "admin"], organizations=["bank-a"], scopes=["信贷"],
                access_level="restricted", reviewed=True, reviewed_by="reviewer",
            ),
        )
        self.last_evidence_request = None

    def active_snapshot(self):
        return "snapshot-1"

    def snapshots(self):
        return [{"snapshot_id": "snapshot-1", "status": "ready", "clause_count": 1}]

    def get_clause(self, evidence_id, user, request, snapshot_id=None):
        self.last_evidence_request = (user, request, snapshot_id)
        metadata = self.clause.metadata
        if (
            evidence_id != self.clause.evidence_id
            or user.role not in metadata.allowed_roles
            or user.organization not in metadata.organizations
            or request.scope not in metadata.scopes
            or not metadata.effective_date <= request.business_date < metadata.expiry_date
        ):
            return None
        return self.clause


class FakeService:
    def __init__(self, *, demo_mode=True, tokens=None):
        self.settings = Settings(
            _env_file=None,
            demo_mode=demo_mode,
            api_auth_tokens_json=SecretStr(json.dumps(tokens or {})),
            llm_api_key=SecretStr("never-expose-this-secret"),
        )
        self.store = FakeStore()
        self.last_query = None
        self.last_ingest = None
        self.activated = None

    def query(self, request, user):
        self.last_query = (request, user)
        return QueryResponse(status="insufficient_evidence", answer="没有足够证据", demo_mode=True)

    def ingest(self, path, metadata):
        assert path.exists()
        self.last_ingest = (path, metadata, path.read_bytes())
        return [self.store.clause]

    def publish(self):
        return "snapshot-1"

    def activate(self, snapshot_id):
        self.activated = snapshot_id


TOKENS = {
    "employee-key": {"user_id": "employee-a", "role": "employee", "organization": "bank-a"},
    "compliance-key": {"user_id": "compliance-a", "role": "compliance", "organization": "bank-a"},
    "admin-key": {"user_id": "admin-a", "role": "admin", "organization": "bank-a"},
}


@pytest.fixture
def service():
    return FakeService(tokens=TOKENS)


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def test_demo_identity_and_health_do_not_expose_secrets(client, service):
    result = client.post("/ask", json={"question": "如何办理业务？"})
    assert result.status_code == 200
    assert service.last_query[1].user_id == "demo-employee"
    health = client.get("/health")
    assert health.json() == {"status": "ready", "mode": "demo", "active_snapshot": "snapshot-1"}
    assert "never-expose" not in health.text


def test_identity_cannot_be_spoofed_and_unknown_tokens_never_fall_back_to_demo(client, service):
    response = client.post("/ask", headers={"X-API-Key": "invalid"}, json={"question": "测试"})
    assert response.status_code == 401
    response = client.post(
        "/ask", headers={"X-API-Key": "employee-key"}, json={"question": "测试", "role": "admin"},
    )
    assert response.status_code == 422
    assert service.last_query is None
    response = client.post("/ask", headers={"X-API-Key": "employee-key"}, json={"question": "测试"})
    assert response.status_code == 200
    assert service.last_query[1].role == "employee"


def test_production_requires_configured_authentication():
    client = TestClient(create_app(FakeService(demo_mode=False)))
    assert client.post("/ask", json={"question": "测试"}).status_code == 503
    configured = TestClient(create_app(FakeService(demo_mode=False, tokens=TOKENS)))
    assert configured.post("/ask", json={"question": "测试"}).status_code == 401
    assert configured.post(
        "/ask", headers={"X-API-Key": "employee-key"}, json={"question": "测试"},
    ).status_code == 200


def test_invalid_identity_configuration_fails_closed(service):
    service.settings.api_auth_tokens_json = SecretStr('{"secret": {"role": "admin"}}')
    client = TestClient(create_app(service))
    response = client.post("/ask", json={"question": "测试"})
    assert response.status_code == 503
    assert "secret" not in response.text


def test_evidence_endpoint_rechecks_server_identity_date_scope_and_snapshot(client, service):
    path = "/evidence/policy-1-v1-article1"
    params = {
        "business_date": "2026-10-11", "scope": "信贷",
        "snapshot_id": "snapshot-1", "include_teaching": True,
    }
    assert client.get(path, params=params, headers={"X-API-Key": "employee-key"}).status_code == 404
    response = client.get(path, params=params, headers={"X-API-Key": "compliance-key"})
    assert response.status_code == 200
    assert response.json()["version"] == "1"
    user, request, snapshot_id = service.store.last_evidence_request
    assert user.user_id == "compliance-a"
    assert request.business_date == date(2026, 10, 11)
    assert request.scope == "信贷"
    assert request.include_teaching is True
    assert snapshot_id == "snapshot-1"
    assert client.get(
        path, params={**params, "business_date": "2027-01-01"}, headers={"X-API-Key": "compliance-key"},
    ).status_code == 404
    assert client.get(
        path, params={**params, "scope": "存款"}, headers={"X-API-Key": "compliance-key"},
    ).status_code == 404
    service.store.last_evidence_request = None
    assert client.get(
        path, params={**params, "snapshot_id": "missing"}, headers={"X-API-Key": "admin-key"},
    ).status_code == 404
    assert service.store.last_evidence_request is None


def test_admin_routes_require_configured_admin_principal(client, service):
    assert client.post("/admin/publish").status_code == 401
    assert client.post("/admin/publish", headers={"X-API-Key": "employee-key"}).status_code == 403
    assert client.get("/admin/snapshots", headers={"X-API-Key": "admin-key"}).status_code == 200
    published = client.post("/admin/publish", headers={"X-API-Key": "admin-key"})
    assert published.json()["snapshot_id"] == "snapshot-1"
    assert client.post(
        "/admin/snapshots/missing/activate", headers={"X-API-Key": "admin-key"},
    ).status_code == 404
    assert service.activated is None
    assert client.post(
        "/admin/snapshots/snapshot-1/activate", headers={"X-API-Key": "admin-key"},
    ).status_code == 200
    assert service.activated == "snapshot-1"


@pytest.mark.parametrize("source_effective_date", [None, date(2024, 6, 1)])
def test_evidence_api_preserves_summary_kind_and_separate_source_effective_date(
    client, service, source_effective_date,
):
    service.store.clause.metadata = service.store.clause.metadata.model_copy(update={
        "content_kind": "summary", "source_effective_date": source_effective_date,
    })
    response = client.get(
        "/evidence/policy-1-v1-article1",
        params={"business_date": "2026-10-12", "scope": "信贷", "snapshot_id": "snapshot-1"},
        headers={"X-API-Key": "compliance-key"},
    )
    assert response.status_code == 200
    evidence = response.json()
    assert evidence["content_kind"] == "summary"
    assert evidence["source_effective_date"] == (
        source_effective_date.isoformat() if source_effective_date else None
    )
    assert evidence["effective_date"] == "2026-01-01"


def test_upload_uses_random_temporary_path_and_authenticated_reviewer(client, service):
    metadata = {
        "policy_id": "upload-policy", "title": "上传制度", "version": "1.0",
        "effective_date": "2026-01-01", "reviewed": True, "reviewed_by": "spoofed",
    }
    response = client.post(
        "/admin/documents", headers={"X-API-Key": "admin-key"},
        data={"metadata_json": json.dumps(metadata)},
        files={"file": ("../../../../sensitive.txt", b"policy content", "text/plain")},
    )
    assert response.status_code == 201
    assert response.json()["status"] == "draft"
    path, recorded_metadata, data = service.last_ingest
    assert isinstance(path, Path)
    assert path.name.startswith("bank-policy-")
    assert "sensitive" not in str(path)
    assert recorded_metadata.reviewed_by == "admin-a"
    assert data == b"policy content"
    assert not path.exists()


def test_upload_limit_rejects_content_before_ingesting(client, service, monkeypatch):
    monkeypatch.setattr("bank_agent.api.MAX_UPLOAD_BYTES", 5)
    response = client.post(
        "/admin/documents", headers={"X-API-Key": "admin-key"},
        data={"metadata_json": json.dumps({
            "policy_id": "too-large", "title": "制度", "version": "1",
            "effective_date": "2026-01-01", "reviewed": True,
        })},
        files={"file": ("policy.txt", b"too large file", "text/plain")},
    )
    assert response.status_code == 413
    assert service.last_ingest is None


def test_internal_failure_details_are_not_returned(client, service):
    def fail(*_args):
        raise RuntimeError("secret-api-key-and-internal-path")

    service.query = fail
    response = client.post("/ask", json={"question": "测试"})
    assert response.status_code == 503
    assert "secret-api-key" not in response.text
