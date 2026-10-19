from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from bank_agent.schemas import DocumentMetadata, QueryRequest, UserContext


def main() -> None:
    parser = argparse.ArgumentParser(description="银行制度 LangGraph Agent（本地管理命令）")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seed-demo", help="导入并发布明确标注的教学样例")
    ingest = commands.add_parser("ingest", help="导入已人工审核文件；随后需 publish")
    ingest.add_argument("file", type=Path)
    ingest.add_argument("--metadata", type=Path, required=True)
    commands.add_parser("publish", help="构建索引并发布知识快照")
    commands.add_parser("snapshots", help="列出知识快照")
    rollback = commands.add_parser("activate", help="切回某个已就绪快照")
    rollback.add_argument("snapshot_id")
    ask = commands.add_parser("ask", help="本地问答；身份参数仅限可信管理员本地测试")
    ask.add_argument("question")
    ask.add_argument("--date", type=date.fromisoformat)
    ask.add_argument("--scope")
    ask.add_argument("--teaching", action="store_true")
    ask.add_argument("--role", choices=["employee", "compliance", "admin"], default="employee")
    ask.add_argument("--organization", default="demo-bank")
    ask.add_argument("--session")
    graph = commands.add_parser("graph", help="导出 LangGraph Mermaid 图")
    graph.add_argument("--output", type=Path, default=Path("docs/agent-graph.mmd"))
    args = parser.parse_args()
    from bank_agent.service import get_service

    service = get_service()
    if args.command == "seed-demo":
        result = {"snapshot_id": service.bootstrap_demo(), "mode": "teaching"}
    elif args.command == "ingest":
        metadata = DocumentMetadata.model_validate_json(args.metadata.read_text(encoding="utf-8"))
        clauses = service.ingest(args.file, metadata)
        result = {"clauses": len(clauses), "evidence_ids": [item.evidence_id for item in clauses],
                  "next": "bank-agent publish"}
    elif args.command == "publish":
        result = {"snapshot_id": service.publish()}
    elif args.command == "snapshots":
        result = service.store.snapshots()
    elif args.command == "activate":
        service.activate(args.snapshot_id)
        result = {"active_snapshot": args.snapshot_id}
    elif args.command == "graph":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(service.agent.graph.get_graph().draw_mermaid(), encoding="utf-8")
        result = {"path": str(args.output.resolve())}
    else:
        values = {"question": args.question, "scope": args.scope, "include_teaching": args.teaching,
                  "session_id": args.session}
        if args.date:
            values["business_date"] = args.date
        result = service.query(QueryRequest(**values), UserContext(
            user_id="local-operator", role=args.role, organization=args.organization
        )).model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
