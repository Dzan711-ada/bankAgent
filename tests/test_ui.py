from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

from bank_agent.config import Settings

UI_PATH = Path(__file__).resolve().parents[1] / "bank_agent" / "ui.py"


def test_ui_starts_and_demo_teaching_is_selected(monkeypatch):
    settings = Settings(_env_file=None, demo_mode=True)
    monkeypatch.setattr("bank_agent.config.get_settings", lambda: settings)
    application = AppTest.from_file(UI_PATH, default_timeout=20).run()
    assert not application.exception
    assert application.title[0].value == "银行规章制度问答"
    assert application.chat_input[0].placeholder == "请输入制度或业务问题"
    teaching = next(widget for widget in application.checkbox if widget.label == "包括教学示例")
    assert teaching.value is True
    assert {tab.label for tab in application.tabs} == {"制度问答", "知识库管理"}


def test_ui_displays_answer_and_evidence_returned_by_api(monkeypatch):
    settings = Settings(_env_file=None, demo_mode=True, api_url="http://127.0.0.1:8000")
    monkeypatch.setattr("bank_agent.config.get_settings", lambda: settings)
    captured_requests = []

    def fake_request(method, url, **kwargs):
        captured_requests.append((method, url, kwargs))
        return httpx.Response(200, json={
            "status": "answered",
            "answer": "须经有权人员审批。[evidence-1]",
            "rationale_summary": "依据第一条中的授权要求。",
            "missing_information": [],
            "applicable_conditions": ["在授权范围内"],
            "evidence": [{
                "evidence_id": "evidence-1", "title": "授权教学制度", "version": "2.0",
                "article": "第一条", "effective_date": "2026-01-01", "text": "须经有权人员审批。",
                "document_type": "teaching", "page": None, "source_url": None,
            }],
            "snapshot_id": "snapshot-1", "session_id": "bound-session", "demo_mode": True,
        })

    monkeypatch.setattr("httpx.request", fake_request)
    application = AppTest.from_file(UI_PATH, default_timeout=20).run()
    application.chat_input[0].set_value("是否需要审批？").run()
    assert not application.exception
    assert len(captured_requests) == 1
    method, url, kwargs = captured_requests[0]
    assert method == "POST" and url.endswith("/ask")
    assert kwargs["json"]["question"] == "是否需要审批？"
    assert kwargs["json"]["include_teaching"] is True
    assert any("须经有权人员审批。[evidence-1]" in item.value for item in application.markdown)
    assert any("版本 2.0" in item.label for item in application.expander)
    assert application.session_state["session_id"] == "bound-session"
    assert any("离线教学演示" in item.value for item in application.warning)


def test_ui_labels_summary_date_and_preserves_unknown_source_effective_date(monkeypatch):
    settings = Settings(_env_file=None, demo_mode=False)
    monkeypatch.setattr("bank_agent.config.get_settings", lambda: settings)

    def fake_request(method, url, **kwargs):
        return httpx.Response(200, json={
            "status": "answered", "answer": "公开资料整理摘要。", "evidence": [{
                "evidence_id": "summary-1", "title": "办卡公开资料摘要", "version": "snapshot-2026.10.12",
                "article": "正文", "effective_date": "2026-10-12", "text": "公开资料整理摘要。",
                "document_type": "public", "content_kind": "summary", "source_effective_date": None,
            }],
        })

    monkeypatch.setattr("httpx.request", fake_request)
    application = AppTest.from_file(UI_PATH, default_timeout=20).run()
    application.chat_input[0].set_value("办卡需要哪些材料？").run()
    assert not application.exception
    assert any("整理摘要（非官方原文）" in item.value for item in application.warning)
    assert any("整理日期：2026-10-12" in item.value for item in application.caption)
    assert any("来源制度生效日期：未确认" in item.value for item in application.caption)
    assert not any("生效日期：2026-10-12" in item.value for item in application.caption)
