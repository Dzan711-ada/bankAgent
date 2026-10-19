"""Server-owned identities; caller input never determines a user's role."""

from __future__ import annotations

import json

from fastapi import HTTPException
from pydantic import ValidationError

from bank_agent.config import Settings
from bank_agent.schemas import UserContext


def resolve_user(settings: Settings, api_key: str | None, *, require_admin: bool = False) -> UserContext:
    raw_tokens = settings.api_auth_tokens_json.get_secret_value().strip()
    try:
        tokens = json.loads(raw_tokens) if raw_tokens else {}
        if not isinstance(tokens, dict):
            raise ValueError("invalid identity map")
        principals = {}
        for token, identity in tokens.items():
            if not isinstance(token, str) or not token.strip() or not isinstance(identity, dict):
                raise ValueError("invalid identity map")
            principals[token] = UserContext.model_validate(identity)
    except (ValueError, TypeError, ValidationError):
        raise HTTPException(status_code=503, detail="服务身份配置无效，请联系管理员") from None

    if not principals and not settings.demo_mode:
        raise HTTPException(status_code=503, detail="服务尚未配置访问凭证")
    if api_key is None:
        if settings.demo_mode and not require_admin:
            return UserContext(user_id="demo-employee", organization="demo-bank")
        raise HTTPException(status_code=401, detail="请提供 X-API-Key 访问凭证")
    user = principals.get(api_key)
    if user is None:
        raise HTTPException(status_code=401, detail="访问凭证无效")
    if require_admin and user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user
