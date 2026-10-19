"""Small transparent regression runner; correctness still requires human review."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from bank_agent.schemas import QueryRequest, UserContext
from bank_agent.service import AppService


def evaluate(service: AppService, dataset: Path) -> dict:
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    snapshot = service.store.active_snapshot()
    for case in cases:
        user = UserContext.model_validate(case.get("user", {"user_id": "evaluation"}))
        request = QueryRequest.model_validate(case["request"])
        start = time.perf_counter()
        response = service.query(request, user)
        latency = (time.perf_counter() - start) * 1000
        retrieved = service.retriever.search(request.question, user, request, snapshot_id=snapshot)
        hit_keys = {(item.clause.metadata.policy_id, item.clause.metadata.version,
                     item.clause.article.split("（片段", 1)[0]) for item in retrieved}
        necessary = {tuple(item) for item in case.get("necessary_evidence", [])}
        recall = len(hit_keys & necessary) / len(necessary) if necessary else None
        used_keys = {(item.title, item.version, item.article) for item in response.evidence}
        rows.append({
            "id": case["id"], "expected_status": case["expected_status"], "status": response.status,
            "status_match": response.status in case["expected_status"], "retrieval_recall": recall,
            "all_necessary_recalled": necessary <= hit_keys if necessary else None,
            "latency_ms": round(latency, 2), "response_evidence_ids": [item.evidence_id for item in response.evidence],
            "response_versions": sorted({key[1] for key in used_keys}), "checks": response.checks,
            "requires_human_review": True,
        })
    recalls = [row["retrieval_recall"] for row in rows if row["retrieval_recall"] is not None]
    latencies = sorted(row["latency_ms"] for row in rows)
    return {
        "snapshot_id": snapshot, "mode": "teaching" if service.settings.demo_mode else "real-model",
        "case_count": len(rows), "retrieval_k": service.settings.evidence_k,
        "status_match_rate": statistics.mean(row["status_match"] for row in rows) if rows else None,
        "mean_necessary_evidence_recall": statistics.mean(recalls) if recalls else None,
        "p50_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": latencies[min(len(latencies) - 1, int(len(latencies) * .95))] if latencies else None,
        "notes": "仅教学回归；状态匹配和召回不代表答案正确率。必要证据必须由业务人员审核。",
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(AppService(), args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已完成 {report['case_count']} 个教学回归案例：{args.output.resolve()}")


if __name__ == "__main__":
    main()
