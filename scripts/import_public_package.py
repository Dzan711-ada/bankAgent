"""Validate a public-source package; import only after a human review."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bank_agent.config import Settings
from bank_agent.knowledge import _read_blocks, _split_clauses
from bank_agent.schemas import DocumentMetadata

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE = WORKSPACE / "knowledge_packages" / "big_four_public_2026_10_12"


def _inside(package: Path, relative: str) -> Path:
    path = (package / relative).resolve()
    if not path.is_relative_to(package) or not path.is_file():
        raise ValueError("资料路径必须是包内已有文件")
    return path


def inspect_package(package: Path) -> tuple[list, dict]:
    manifest = json.loads((package / "sources.json").read_text(encoding="utf-8"))
    parser_settings = Settings(_env_file=None, demo_mode=True)
    documents, report = [], []
    for entry in manifest["entries"]:
        policy = _inside(package, entry["policy_file"])
        metadata_path = _inside(package, entry["metadata_file"])
        for path, checksum in ((policy, "policy_sha256"), (metadata_path, "metadata_sha256")):
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry[checksum]:
                raise ValueError("资料已变化，请先核对内容、版本号和 sources.json 中的校验值")
        metadata = DocumentMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        if metadata.content_kind != "summary" or metadata.document_type != "public":
            raise ValueError("本包必须明确标为公开资料摘要")
        if metadata.reviewed or metadata.reviewed_by:
            raise ValueError("原始整理包必须保持待人工审核，不得预置审核身份")
        if metadata.organizations or metadata.scopes != [entry["scope"]]:
            raise ValueError("公开资料不得用银行名称冒充用户机构权限；范围须匹配银行 scope")
        if metadata.source_url != entry["source_url"]:
            raise ValueError("正文来源与元数据不一致")
        clauses = _split_clauses(_read_blocks(policy), metadata, parser_settings)
        if not clauses:
            raise ValueError("资料未解析到正文")
        documents.append((entry, policy, metadata))
        report.append({"id": entry["id"], "scope": entry["scope"], "clauses": len(clauses),
                       "content_kind": metadata.content_kind, "reviewed": metadata.reviewed,
                       "source_effective_date": entry["original_effective_date"]})
    return documents, {"status": "format_valid_pending_review", "documents": report,
                       "document_count": len(report), "clause_count": sum(r["clauses"] for r in report),
                       "network_calls": 0, "knowledge_base_modified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="四大行公开资料包：默认仅检查，不加载 .env、不调用模型接口")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--check", action="store_true", help="仅校验格式和条款解析（默认）")
    parser.add_argument("--import-reviewed", action="store_true", help="人工核对资料后导入草稿")
    parser.add_argument("--reviewed-by", default="", help="真实审核人或工号")
    parser.add_argument("--confirm-rights", action="store_true", help="确认已核验用途与所需授权")
    parser.add_argument("--rights-basis", default="", help="记录实际使用依据、授权编号或核验结论")
    parser.add_argument("--publish", action="store_true", help="导入后建立云向量索引并发布")
    args = parser.parse_args()
    if args.check and args.import_reviewed:
        parser.error("--check 与 --import-reviewed 不可同时使用")
    if args.publish and not args.import_reviewed:
        parser.error("--publish 必须与 --import-reviewed 同时使用")
    if args.import_reviewed and not (args.reviewed_by.strip() and args.confirm_rights
                                     and args.rights_basis.strip()):
        parser.error("导入前需要真实审核人、--confirm-rights 和实际 --rights-basis；请先阅读包内 README")
    package = args.package.resolve()
    documents, report = inspect_package(package)
    if not args.import_reviewed:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    from bank_agent.service import get_service

    service = get_service()
    approved_dir = package / "reviewed"
    approved_dir.mkdir(exist_ok=True)
    audit = {"reviewed_by": args.reviewed_by.strip(), "rights_basis": args.rights_basis.strip(),
             "reviewed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "imports": []}
    for entry, policy, metadata in documents:
        approved = metadata.model_copy(update={"reviewed": True, "reviewed_by": args.reviewed_by.strip()})
        clauses = service.ingest(policy, approved)
        (approved_dir / f"{entry['id']}.json").write_text(
            approved.model_dump_json(indent=2), encoding="utf-8")
        audit["imports"].append({"policy_id": approved.policy_id, "version": approved.version,
                                 "clauses": len(clauses), "policy_sha256": entry["policy_sha256"]})
    (approved_dir / "review_record.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    output = {"status": "draft_imported", "documents": len(documents)}
    if args.publish:
        output.update(status="published", snapshot_id=service.publish())
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Provider exceptions can contain credentials; show only the class.
        raise SystemExit(f"资料包处理失败（{type(exc).__name__}）；请核对包内文件、权限与服务配置。") from None
