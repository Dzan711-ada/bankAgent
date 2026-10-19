import json
import runpy
import sys
from pathlib import Path

import pytest


def _importer():
    workspace = Path(__file__).resolve().parent.parent
    return runpy.run_path(str(workspace / "scripts" / "import_public_package.py"))


def test_public_package_check_never_initializes_live_model_service(monkeypatch, capsys):
    def forbidden():
        pytest.fail("格式检查不应初始化带模型凭证的业务服务")

    monkeypatch.setattr("bank_agent.service.get_service", forbidden)
    monkeypatch.setattr(sys, "argv", ["import_public_package.py", "--check"])
    module = _importer()
    module["main"]()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "format_valid_pending_review"
    assert report["document_count"] == 7 and report["clause_count"] == 28
    assert report["network_calls"] == 0 and not report["knowledge_base_modified"]
    assert all(not item["reviewed"] and item["content_kind"] == "summary" for item in report["documents"])
    assert sum(item["source_effective_date"] is None for item in report["documents"]) == 2


def test_public_package_import_without_use_rights_review_stops_before_service(monkeypatch):
    def forbidden():
        pytest.fail("未经用途及授权核验不应触发知识库导入")

    monkeypatch.setattr("bank_agent.service.get_service", forbidden)
    monkeypatch.setattr(sys, "argv", ["import_public_package.py", "--import-reviewed",
                                     "--reviewed-by", "reviewer-test"])
    with pytest.raises(SystemExit) as error:
        _importer()["main"]()
    assert error.value.code == 2
