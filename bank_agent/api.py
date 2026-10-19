from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import ValidationError

from bank_agent.auth import resolve_user
from bank_agent.schemas import (
    DocumentMetadata,
    EvidenceView,
    QueryRequest,
    QueryResponse,
    UserContext,
    business_today,
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def get_api_service():
    # Construction is lazy so importing the ASGI app does not call external APIs.
    from bank_agent.service import get_service

    try:
        return get_service()
    except Exception:
        raise HTTPException(status_code=503, detail="服务初始化失败，请检查服务端配置") from None


def current_user(
    service: Annotated[Any, Depends(get_api_service)],
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> UserContext:
    return resolve_user(service.settings, api_key)


def admin_user(
    service: Annotated[Any, Depends(get_api_service)],
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> UserContext:
    return resolve_user(service.settings, api_key, require_admin=True)


def _evidence_view(clause) -> EvidenceView:
    metadata = clause.metadata
    return EvidenceView(
        evidence_id=clause.evidence_id,
        title=metadata.title,
        version=metadata.version,
        article=clause.article,
        effective_date=metadata.effective_date,
        text=clause.text,
        document_type=metadata.document_type,
        content_kind=metadata.content_kind,
        source_effective_date=metadata.source_effective_date,
        page=clause.page,
        source_url=metadata.source_url,
    )


def create_app(service=None) -> FastAPI:
    application = FastAPI(title="银行规章制度问答 Agent", version="0.1.0")
    if service is not None:
        application.dependency_overrides[get_api_service] = lambda: service

    @application.get("/health")
    def health(service: Annotated[Any, Depends(get_api_service)]):
        return {
            "status": "ready",
            "mode": "demo" if service.settings.demo_mode else "production",
            "active_snapshot": service.store.active_snapshot(),
        }

    @application.post("/ask", response_model=QueryResponse)
    def ask(
        request: QueryRequest,
        user: Annotated[UserContext, Depends(current_user)],
        service: Annotated[Any, Depends(get_api_service)],
    ):
        try:
            return service.query(request, user)
        except ValueError:
            raise HTTPException(status_code=422, detail="问题或会话参数无效") from None
        except Exception:
            raise HTTPException(status_code=503, detail="问答处理失败，请稍后重试或联系管理员") from None

    @application.get("/evidence/{evidence_id}", response_model=EvidenceView)
    def evidence(
        evidence_id: str,
        user: Annotated[UserContext, Depends(current_user)],
        service: Annotated[Any, Depends(get_api_service)],
        business_date: str | None = None,
        include_teaching: bool = False,
        scope: str | None = None,
        snapshot_id: str | None = None,
    ):
        try:
            request = QueryRequest(
                question="证据查阅",
                business_date=business_date or business_today(),
                include_teaching=include_teaching,
                scope=scope,
            )
        except ValidationError:
            raise HTTPException(status_code=422, detail="业务日期或适用范围无效") from None
        try:
            if snapshot_id and not any(
                snapshot.get("snapshot_id", snapshot.get("id")) == snapshot_id
                for snapshot in service.store.snapshots()
            ):
                raise HTTPException(status_code=404, detail="证据不存在或无权访问")
            clause = service.store.get_clause(evidence_id, user, request, snapshot_id=snapshot_id)
        except HTTPException:
            raise
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="证据不存在或无权访问") from None
        except Exception:
            raise HTTPException(status_code=503, detail="证据查询暂不可用") from None
        if clause is None:
            raise HTTPException(status_code=404, detail="证据不存在或无权访问")
        return _evidence_view(clause)

    @application.post("/admin/documents", status_code=201)
    async def ingest_document(
        user: Annotated[UserContext, Depends(admin_user)],
        service: Annotated[Any, Depends(get_api_service)],
        file: Annotated[UploadFile, File()],
        metadata_json: Annotated[str, Form()],
    ):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=422, detail="支持 PDF、DOCX、TXT 和 Markdown 文件")
        if len(metadata_json) > 32_768:
            raise HTTPException(status_code=422, detail="文档元数据过长")
        try:
            values = json.loads(metadata_json)
            if not isinstance(values, dict):
                raise ValueError("metadata must be an object")
            if values.get("reviewed") is True:
                values["reviewed_by"] = user.user_id
            metadata = DocumentMetadata.model_validate(values)
        except (ValueError, TypeError, ValidationError):
            raise HTTPException(status_code=422, detail="文档元数据无效，请检查字段、日期与审核信息") from None
        temporary_path = None
        try:
            with NamedTemporaryFile(prefix="bank-policy-", suffix=suffix, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                size = 0
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="文件不能超过 10 MB")
                    temporary.write(chunk)
            if not size:
                raise HTTPException(status_code=422, detail="文件内容为空")
            # The only filename used on disk is generated by NamedTemporaryFile.
            # Import creates a draft; publishing is a separate authenticated action.
            from starlette.concurrency import run_in_threadpool

            clauses = await run_in_threadpool(service.ingest, temporary_path, metadata)
            return {"status": "draft", "policy_id": metadata.policy_id, "clauses": len(clauses)}
        except HTTPException:
            raise
        except ValueError:
            raise HTTPException(status_code=422, detail="文档解析失败或内容不符合导入要求") from None
        except Exception:
            raise HTTPException(status_code=503, detail="文档导入失败，请检查服务端配置") from None
        finally:
            await file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @application.post("/admin/publish")
    def publish(
        user: Annotated[UserContext, Depends(admin_user)],
        service: Annotated[Any, Depends(get_api_service)],
    ):
        try:
            return {"snapshot_id": service.publish()}
        except ValueError:
            raise HTTPException(status_code=422, detail="草稿尚不能发布，请检查审核与文档版本") from None
        except Exception:
            raise HTTPException(status_code=503, detail="知识库发布失败，请检查服务端日志") from None

    @application.get("/admin/snapshots")
    def snapshots(
        user: Annotated[UserContext, Depends(admin_user)],
        service: Annotated[Any, Depends(get_api_service)],
    ):
        try:
            return {
                "snapshots": service.store.snapshots(),
                "active_snapshot": service.store.active_snapshot(),
            }
        except Exception:
            raise HTTPException(status_code=503, detail="知识库快照查询暂不可用") from None

    @application.post("/admin/snapshots/{snapshot_id}/activate")
    def activate(
        snapshot_id: str,
        user: Annotated[UserContext, Depends(admin_user)],
        service: Annotated[Any, Depends(get_api_service)],
    ):
        try:
            if not any(
                snapshot.get("snapshot_id", snapshot.get("id")) == snapshot_id
                for snapshot in service.store.snapshots()
            ):
                raise HTTPException(status_code=404, detail="知识库快照不存在")
            service.activate(snapshot_id)
            return {"snapshot_id": snapshot_id, "status": "active"}
        except HTTPException:
            raise
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="知识库快照不存在或不可用") from None
        except Exception:
            raise HTTPException(status_code=503, detail="知识库快照切换失败") from None

    return application


app = create_app()

