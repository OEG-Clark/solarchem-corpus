from typing import Any, Dict


def _clean_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _require_api_key(api_key: Any, provider: str) -> str:
    if not api_key or "REPLACE-ME" in str(api_key):
        raise ValueError(
            f"`model.{provider}.api_key` is missing or still set to the "
            f"placeholder. Please put a real key in config.yaml."
        )
    return str(api_key)


def get_model_id(model_cfg: Dict[str, Any]) -> str:
    provider = (model_cfg.get("provider") or "").lower().strip()
    if not provider:
        raise ValueError("`model.provider` is required in config.yaml")
    provider_cfg = model_cfg.get(provider) or {}
    model_id = provider_cfg.get("id")
    if not model_id:
        raise ValueError(f"`model.{provider}.id` is required in config.yaml")
    return str(model_id)


def build_model(model_cfg: Dict[str, Any]):
    provider = (model_cfg.get("provider") or "").lower().strip()
    if not provider:
        raise ValueError("`model.provider` is required in config.yaml")

    provider_cfg = model_cfg.get(provider) or {}
    model_id = provider_cfg.get("id")
    if not model_id:
        raise ValueError(f"`model.{provider}.id` is required in config.yaml")

    shared = {
        "temperature": model_cfg.get("temperature"),
        "max_tokens": model_cfg.get("max_tokens"),
    }

    if provider == "deepseek":
        from agno.models.deepseek import DeepSeek

        api_key = _require_api_key(provider_cfg.get("api_key"), "deepseek")
        kwargs = _clean_kwargs({
            "id": model_id,
            "api_key": api_key,
            **shared,
        })
        return DeepSeek(**kwargs)

    if provider == "ollama":
        from agno.models.ollama import Ollama

        kwargs = _clean_kwargs({
            "id": model_id,
            "host": provider_cfg.get("host"),
            "temperature": shared["temperature"],
        })
        return Ollama(**kwargs)

    if provider == "openrouter":
        from agno.models.openrouter import OpenRouter

        api_key = _require_api_key(provider_cfg.get("api_key"), "openrouter")
        kwargs = _clean_kwargs({
            "id": model_id,
            "api_key": api_key,
            **shared,
        })
        return OpenRouter(**kwargs)

    raise ValueError(
        f"Unknown provider '{provider}'. Supported: deepseek, ollama, openrouter."
    )
