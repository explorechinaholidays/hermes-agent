"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

from typing import Any


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``key_env`` names the env var
    holding the key; ``api_key_env`` accepted as an alias). Returns None when
    neither yields a non-empty value, letting ``resolve_runtime_provider``
    fall through to the provider's standard credential resolution.

    ``key_env`` is resolved through ``agent.secret_scope.get_secret`` rather
    than a raw ``os.getenv`` — in a multiplexed gateway a bare env read would
    ignore the active profile's scope and can return another profile's
    credential. ``get_secret`` already implements the right fallback: it
    reads ``os.environ`` when there's no active multiplexed scope (matching
    prior single-profile behavior), and fails closed only when multiplexing
    is active with no scope installed.
    """
    if not isinstance(entry, dict):
        return None
    inline = str(entry.get("api_key") or "").strip()
    if inline:
        return inline
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if key_env:
        from agent.secret_scope import get_secret

        return (get_secret(key_env) or "").strip() or None
    return None


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def entry_matches_model(entry: dict[str, Any], primary_model: str | None) -> bool:
    """True when *entry* applies to the primary model *primary_model*.

    Entries may carry an optional ``for_models`` list restricting which
    primary models they serve (per-model fallback tiering)::

        fallback_providers:
          - provider: custom:qwen
            model: qwen3.8-max
            for_models: ["glm-5.3"]        # only when glm-5.3 is primary
          - provider: custom:qwen
            model: qwen3.8-flash
            for_models: ["glm-5.3-flash"]

    Match rules per pattern (case-insensitive):

    - ``"glm-5.3*"`` — trailing ``*`` is a prefix wildcard: matches
      ``glm-5.3`` and ``glm-5.3-flash`` alike.
    - ``"glm-5.3"`` — exact match only.

    Back-compat: an entry WITHOUT ``for_models`` (or with an empty /
    malformed one — no valid non-empty string patterns) is GLOBAL and
    matches every primary. Fallback is a resilience mechanism, so a
    restriction that parses to nothing fails open toward "applies" —
    an entry must never silently drop out of the chain because of a
    typo'd key.
    """
    if not primary_model:
        return True
    raw = entry.get("for_models")
    if not isinstance(raw, (list, tuple)):
        # Absent key (or a scalar/garbage value) → global entry.
        return True
    patterns = [
        p.strip().lower()
        for p in raw
        if isinstance(p, str) and p.strip()
    ]
    if not patterns:
        # ``for_models: []`` (or all-empty) → no usable restriction.
        return True
    primary = primary_model.strip().lower()
    if not primary:
        return True
    for pattern in patterns:
        if pattern.endswith("*"):
            if primary.startswith(pattern[:-1]):
                return True
        elif primary == pattern:
            return True
    return False


def get_fallback_chain(
    config: dict[str, Any] | None,
    primary_model: str | None = None,
) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies.

    When *primary_model* is given, entries carrying ``for_models`` are
    filtered to those whose patterns match the primary (see
    ``entry_matches_model``); entries without ``for_models`` stay global.
    Omitting *primary_model* returns the unfiltered chain (legacy behavior —
    call sites that don't know the primary, e.g. presence checks, keep
    working unchanged).
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            if not entry_matches_model(entry, primary_model):
                continue
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain
