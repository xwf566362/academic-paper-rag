# -*- coding: utf-8 -*-
"""Configuration API - set provider + API key at runtime."""
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import yaml

router = APIRouter(prefix="/api/config", tags=["config"])

class ProviderConfig(BaseModel):
    provider: str
    api_key: str
    base_url: str = ""
    model: str = ""

@router.post("/provider")
async def set_provider(cfg: ProviderConfig):
    """Set a provider's API key and config at runtime."""
    from app.config import load_config, get_config

    app_cfg = get_config()
    if cfg.provider not in app_cfg.llm_providers:
        raise HTTPException(status_code=404, detail=f"Unknown provider: {cfg.provider}")

    # Update in-memory config
    provider_cfg = app_cfg.llm_providers[cfg.provider]
    provider_cfg.api_key = cfg.api_key
    if cfg.base_url:
        provider_cfg.base_url = cfg.base_url
    if cfg.model:
        provider_cfg.model = cfg.model

    # Clear LLM client cache so next request picks up new config
    from app.api.llm_client import _clients
    _clients.pop(cfg.provider, None)

    # Persist to local config file
    cfg_dir = Path(__file__).resolve().parent.parent
    local_cfg_path = cfg_dir / "config.local.yaml"
    local = {"llm_providers": {cfg.provider: {"api_key": cfg.api_key}}}
    if cfg.base_url:
        local["llm_providers"][cfg.provider]["base_url"] = cfg.base_url
    if cfg.model:
        local["llm_providers"][cfg.provider]["model"] = cfg.model
    if local_cfg_path.exists():
        with open(local_cfg_path, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        existing.setdefault("llm_providers", {}).setdefault(cfg.provider, {}).update(local["llm_providers"][cfg.provider])
        local = existing
    with open(local_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(local, f, allow_unicode=True)

    return {"status": "ok", "provider": cfg.provider}

@router.get("/providers")
async def list_providers():
    """List available providers."""
    from app.config import get_config
    cfg = get_config()
    return {
        "default": cfg.default_provider,
        "providers": {
            name: {
                "base_url": p.base_url,
                "model": p.model,
                "configured": bool(p.api_key) and not p.api_key.startswith("your-"),
            }
            for name, p in cfg.llm_providers.items()
        },
    }
