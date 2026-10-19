from datetime import date

import pytest

from bank_agent.config import Settings
from bank_agent.schemas import QueryRequest, UserContext
from bank_agent.service import AppService


@pytest.fixture(scope="module")
def demo_service(tmp_path_factory):
    service = AppService(Settings(_env_file=None, data_dir=tmp_path_factory.mktemp("integration")))
    service.bootstrap_demo()
    return service


def test_full_demo_pipeline_historical_and_current_versions(demo_service):
    user = UserContext(user_id="employee")
    for when, version, extra_check in [(date(2025, 6, 1), "1.0", False), (date(2026, 6, 1), "2.0", True)]:
        result = demo_service.query(QueryRequest(
            question="生产环境提取客户数据有哪些审批要求？", business_date=when,
            include_teaching=True, scope="data",
        ), user)
        assert result.status == "answered", result.model_dump()
        assert result.claims
        assert {evidence.version for evidence in result.evidence} == {version}
        assert ("数据管理专岗复核" in result.answer) is extra_check
        assert all(claim.text == "\n".join(claim.supporting_quotes) for claim in result.claims)


def test_full_demo_does_not_leak_restricted_clauses(demo_service):
    request = QueryRequest(question="特殊授权申请超过500万元谁审批？", business_date=date(2026, 6, 1),
                           include_teaching=True, scope="authorization")
    result = demo_service.query(request, UserContext(user_id="employee"))
    assert result.status == "insufficient_evidence"
    assert result.evidence == []
    assert "500万元" not in result.answer
    authorized = demo_service.query(request, UserContext(user_id="compliance", role="compliance"))
    assert authorized.status == "answered"
    assert "授权委员会审批" in authorized.answer


def test_full_cross_clause_case_preserves_all_conditions(demo_service):
    result = demo_service.query(QueryRequest(
        question="主管已经同意，我能否从生产系统导出客户明细交给外部合作方做系统演示？",
        business_date=date(2026, 6, 1), include_teaching=True, scope="data",
    ), UserContext(user_id="employee"))
    assert result.status == "clarification_needed", result.model_dump()
    for phrase in ["业务部门负责人", "安全评估", "匿名化"]:
        assert phrase in result.answer
    assert result.missing_information


def test_demo_teaching_needs_explicit_optin(demo_service):
    result = demo_service.query(QueryRequest(question="数据提取需要哪些审批？"), UserContext(user_id="employee"))
    assert result.status == "insufficient_evidence"
    assert result.evidence == []
