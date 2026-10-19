"""Streamlit client for the authenticated API; it does not access the store directly."""

from __future__ import annotations

import json
from datetime import date
from uuid import uuid4

import httpx
import streamlit as st

from bank_agent.config import get_settings
from bank_agent.schemas import business_today

STATUS_LABELS = {
    "answered": "已找到制度依据",
    "clarification_needed": "需要补充业务信息",
    "insufficient_evidence": "目前依据不足",
    "review_required": "需要人工复核",
}


def api_call(method: str, path: str, api_url: str, api_key: str, **kwargs):
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        response = httpx.request(method, api_url.rstrip("/") + path, headers=headers, timeout=120, **kwargs)
        payload = response.json()
        if not response.is_success:
            detail = payload.get("detail", "请求失败")
            if not isinstance(detail, str):
                detail = "输入参数无效，请检查后重试"
            st.error(f"{response.status_code}：{detail}")
            return None
        return payload
    except (httpx.HTTPError, ValueError):
        st.error("无法连接问答服务。请先启动 API 服务，并检查 .env 中的 API_URL。")
        return None


def render_answer(answer: dict):
    label = STATUS_LABELS.get(answer.get("status"), "处理结果")
    st.caption(label)
    if answer.get("demo_mode"):
        st.warning("当前为离线教学演示。结果使用示例材料，不能作为实际业务依据。")
    st.markdown(answer.get("answer", ""))
    if answer.get("missing_information"):
        st.info("请补充：" + "；".join(answer["missing_information"]))
    if answer.get("applicable_conditions"):
        st.write("适用条件：" + "；".join(answer["applicable_conditions"]))
    if answer.get("rationale_summary"):
        with st.expander("业务依据摘要"):
            st.write(answer["rationale_summary"])
    for evidence in answer.get("evidence", []):
        label = f"{evidence['title']} · {evidence['article']} · 版本 {evidence['version']}"
        with st.expander(label):
            is_summary = evidence.get("content_kind") == "summary"
            date_label = "整理日期" if is_summary else "生效日期"
            st.caption(
                f"{date_label}：{evidence['effective_date']} ｜ "
                f"证据编号：{evidence['evidence_id']}"
                + (f" ｜ 页码：{evidence['page']}" if evidence.get("page") else "")
            )
            if evidence.get("document_type") == "teaching":
                st.warning("教学示例材料")
            if is_summary:
                st.warning("整理摘要（非官方原文），请通过来源链接核对官方资料。")
                st.caption("来源制度生效日期：" + (evidence.get("source_effective_date") or "未确认"))
            st.text(evidence.get("text", ""))
            source_url = evidence.get("source_url")
            if source_url and source_url.startswith(("https://", "http://")):
                st.link_button("打开原始来源", source_url)
    if answer.get("snapshot_id"):
        st.caption(f"知识库版本：{answer['snapshot_id']}")


def render_admin(api_url: str, api_key: str):
    st.write("管理员凭证由服务端配置。导入文件经过人工审核后进入草稿库，发布后才用于问答。")
    content_label = st.selectbox("内容类型", ["原文", "整理摘要（非官方原文）"])
    content_kind = "summary" if content_label == "整理摘要（非官方原文）" else "original"
    with st.form("document_import"):
        file = st.file_uploader("制度文件（最多 10 MB）", type=["pdf", "docx", "txt", "md"])
        columns = st.columns(2)
        policy_id = columns[0].text_input("制度 ID", placeholder="credit-policy")
        version = columns[1].text_input("版本", value="1.0")
        title = st.text_input("制度名称")
        effective_date = st.date_input(
            "整理日期" if content_kind == "summary" else "生效日期", value=business_today(), key="effective_date"
        )
        source_effective_text = ""
        if content_kind == "summary":
            st.caption("整理日期仅表示摘要可用于本知识库的日期；来源制度生效日期需另行核实。")
            source_effective_text = st.text_input("来源制度生效日期（可选，YYYY-MM-DD；未知留空）")
        document_type = st.selectbox("资料类型", ["internal", "public", "teaching"])
        access_level = st.selectbox("访问级别", ["internal", "public", "restricted"])
        scope = st.text_input("适用范围（逗号分隔；留空表示不限范围）")
        organizations = st.text_input("适用机构（逗号分隔；留空表示不限机构）")
        roles = st.multiselect(
            "允许访问的角色", ["employee", "compliance", "admin"],
            default=["employee", "compliance", "admin"],
        )
        issuer = st.text_input("发文机构")
        document_number = st.text_input("文号")
        source_url = st.text_input("原始来源链接（可选）")
        expire = st.checkbox("设置废止日期")
        expiry_date = st.date_input("废止日期", value=business_today(), key="expiry_date")
        reviewed = st.checkbox("我已人工核对正文、版本、生效日期及访问权限")
        submitted = st.form_submit_button("导入已审核草稿")
    if submitted:
        if not file or not policy_id or not title or not roles or not reviewed:
            st.error("请上传文件，填写制度 ID 和名称，选择可访问角色，并完成人工审核。")
        else:
            try:
                source_effective_date = date.fromisoformat(source_effective_text.strip()) if (
                    source_effective_text.strip()) else None
            except ValueError:
                st.error("来源制度生效日期需填写 YYYY-MM-DD；未知请留空，不能用发布日期代替。")
                return
            metadata = {
                "policy_id": policy_id,
                "title": title,
                "version": version,
                "effective_date": effective_date.isoformat(),
                "expiry_date": expiry_date.isoformat() if expire else None,
                "issuer": issuer,
                "document_number": document_number,
                "source_url": source_url or None,
                "access_level": access_level,
                "allowed_roles": roles,
                "organizations": [
                    value.strip() for value in organizations.split(",") if value.strip()
                ],
                "scopes": [value.strip() for value in scope.split(",") if value.strip()],
                "document_type": document_type,
                "content_kind": content_kind,
                "source_effective_date": source_effective_date.isoformat() if source_effective_date else None,
                "reviewed": reviewed,
            }
            result = api_call(
                "POST", "/admin/documents", api_url, api_key,
                files={"file": (file.name, file.getvalue(), file.type)},
                data={"metadata_json": json.dumps(metadata, ensure_ascii=False)},
            )
            if result:
                st.success(f"已导入草稿：{result['clauses']} 条证据。")
    if st.button("发布已审核草稿", type="primary"):
        with st.spinner("正在建立并核验知识库…"):
            result = api_call("POST", "/admin/publish", api_url, api_key)
        if result:
            st.success(f"已发布知识库版本：{result['snapshot_id']}")
    if st.button("查看知识库版本"):
        result = api_call("GET", "/admin/snapshots", api_url, api_key)
        if result:
            st.session_state["snapshots"] = result
    snapshots = st.session_state.get("snapshots", {})
    if snapshots:
        st.caption(f"当前版本：{snapshots.get('active_snapshot', '')}")
        st.dataframe(snapshots.get("snapshots", []), use_container_width=True)
        ids = [
            item["snapshot_id"] for item in snapshots.get("snapshots", [])
            if item.get("status") == "ready"
        ]
        if ids:
            selected = st.selectbox("切换至可用版本", ids)
            if st.button("激活选定版本"):
                result = api_call("POST", f"/admin/snapshots/{selected}/activate", api_url, api_key)
                if result:
                    st.session_state.pop("snapshots", None)
                    st.success(f"已切换至：{selected}")


def main():
    st.set_page_config(page_title="银行规章制度问答", page_icon="🏦", layout="wide")
    settings = get_settings()
    api_url = settings.api_url
    st.title("银行规章制度问答")
    st.caption("按业务日期、适用范围和当前用户权限检索制度，并核对回答的条款依据。")
    with st.sidebar:
        st.caption(f"API 服务：{api_url}")
        api_key = st.text_input("访问凭证", type="password", key="api_key")
        st.caption("凭证对应的身份与机构由服务端确定。演示模式可留空；管理操作需要管理员凭证。")
        business_date = st.date_input("业务日期", value=business_today())
        scope = st.text_input("适用范围", placeholder="例如：个人贷款")
        include_teaching = st.checkbox("包括教学示例", value=settings.demo_mode)
        if st.button("清空会话"):
            st.session_state["messages"] = []
            st.session_state["session_id"] = uuid4().hex
            st.session_state.pop("snapshots", None)
            st.rerun()
    # A credential change starts a fresh chat to avoid showing another identity's previous evidence.
    if st.session_state.get("identity_key") != api_key:
        st.session_state["messages"] = []
        st.session_state["session_id"] = uuid4().hex
        st.session_state.pop("snapshots", None)
        st.session_state["identity_key"] = api_key
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", uuid4().hex)
    question_tab, admin_tab = st.tabs(["制度问答", "知识库管理"])
    with question_tab:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    render_answer(message["content"])
        question = st.chat_input("请输入制度或业务问题")
        if question:
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            request = {
                "question": question,
                "business_date": business_date.isoformat(),
                "scope": scope.strip() or None,
                "include_teaching": include_teaching,
                "session_id": st.session_state["session_id"],
            }
            with st.chat_message("assistant"):
                with st.spinner("正在检索制度并核验依据…"):
                    answer = api_call("POST", "/ask", api_url, api_key, json=request)
                if answer:
                    st.session_state["session_id"] = answer.get("session_id") or st.session_state["session_id"]
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    render_answer(answer)
    with admin_tab:
        render_admin(api_url, api_key)


if __name__ == "__main__":
    main()
