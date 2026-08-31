"""Per-model fallback tiering — ``for_models:`` on fallback chain entries.

EXP-300 / T11. ``entry_matches_model`` owns the match rules; these tests pin:

- absent/malformed/empty ``for_models`` → GLOBAL (back-compat, fail-open)
- exact match and trailing-``*`` prefix match, case-insensitive
- ``get_fallback_chain(primary_model=...)`` filters at consumption time
- no primary → unfiltered (legacy callers unchanged)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_cli.fallback_config import entry_matches_model, get_fallback_chain


def _entry(model="qwen3.8-max", for_models=None, provider="custom:qwen", **extra):
    e = {"provider": provider, "model": model}
    if for_models is not None:
        e["for_models"] = for_models
    e.update(extra)
    return e


class TestEntryMatchesModel:
    def test_absent_for_models_is_global(self):
        assert entry_matches_model(_entry(), "glm-5.3")
        assert entry_matches_model(_entry(), "glm-5.3-flash")
        assert entry_matches_model(_entry(), None)

    def test_none_primary_is_global(self):
        # Callers that don't know the primary (presence checks) must see
        # every entry — legacy behaviour preserved.
        assert entry_matches_model(_entry(for_models=["glm-5.3"]), None)

    def test_malformed_for_models_fails_open(self):
        # A restriction that parses to nothing must never silently drop an
        # entry out of the resilience chain.
        assert entry_matches_model(_entry(for_models="glm-5.3"), "glm-5.3")
        assert entry_matches_model(_entry(for_models=["", "  "]), "glm-5.3")
        assert entry_matches_model(_entry(for_models=[]), "glm-5.3")
        assert entry_matches_model(_entry(for_models=[42, None]), "glm-5.3")

    def test_exact_match_case_insensitive(self):
        e = _entry(for_models=["GLM-5.3"])
        assert entry_matches_model(e, "glm-5.3")
        assert not entry_matches_model(e, "glm-5.3-flash")

    def test_prefix_wildcard(self):
        e = _entry(for_models=["glm-5.3*"])
        assert entry_matches_model(e, "glm-5.3")
        assert entry_matches_model(e, "glm-5.3-flash")
        assert not entry_matches_model(e, "qwen3.8-flash")

    def test_primary_whitespace_tolerated(self):
        e = _entry(for_models=["glm-5.3"])
        assert entry_matches_model(e, "  glm-5.3  ")

    def test_owner_tiering_pairs(self):
        # The exact owner tiering this ticket implements.
        max_e = _entry(model="qwen3.8-max", for_models=["glm-5.3"])
        flash_e = _entry(model="qwen3.8-flash", for_models=["glm-5.3-flash"])
        assert entry_matches_model(max_e, "glm-5.3")
        assert not entry_matches_model(max_e, "glm-5.3-flash")
        assert entry_matches_model(flash_e, "glm-5.3-flash")
        assert not entry_matches_model(flash_e, "glm-5.3")


class TestGetFallbackChainFiltering:
    def _cfg(self):
        return {
            "fallback_providers": [
                _entry(model="qwen3.8-max", for_models=["glm-5.3"]),
                _entry(model="qwen3.8-flash", for_models=["glm-5.3-flash"]),
            ]
        }

    def test_filter_glm53_sees_max_only(self):
        chain = get_fallback_chain(self._cfg(), primary_model="glm-5.3")
        assert [e["model"] for e in chain] == ["qwen3.8-max"]

    def test_filter_glm53_flash_sees_flash_only(self):
        chain = get_fallback_chain(self._cfg(), primary_model="glm-5.3-flash")
        assert [e["model"] for e in chain] == ["qwen3.8-flash"]

    def test_no_primary_returns_unfiltered(self):
        chain = get_fallback_chain(self._cfg())
        assert [e["model"] for e in chain] == ["qwen3.8-max", "qwen3.8-flash"]

    def test_global_entry_applies_to_every_primary(self):
        cfg = {
            "fallback_providers": [
                _entry(model="qwen3.8-max"),  # global
                _entry(model="qwen3.8-flash", for_models=["glm-5.3-flash"]),
            ]
        }
        assert [e["model"] for e in get_fallback_chain(cfg, primary_model="glm-5.3")] == ["qwen3.8-max"]
        assert [e["model"] for e in get_fallback_chain(cfg, primary_model="glm-5.3-flash")] == [
            "qwen3.8-max",
            "qwen3.8-flash",
        ]

    def test_order_preserved_and_copies_fresh(self):
        chain = get_fallback_chain(self._cfg(), primary_model="glm-5.3")
        chain[0]["model"] = "mutated"
        assert get_fallback_chain(self._cfg(), primary_model="glm-5.3")[0]["model"] == "qwen3.8-max"

    def test_legacy_fallback_model_filtered_too(self):
        cfg = {
            "fallback_model": _entry(model="qwen3.8-max", for_models=["glm-5.3"]),
        }
        assert get_fallback_chain(cfg, primary_model="glm-5.3-flash") == []
        assert [e["model"] for e in get_fallback_chain(cfg, primary_model="glm-5.3")] == ["qwen3.8-max"]

    def test_dup_across_keys_still_deduped_after_filter(self):
        cfg = {
            "fallback_providers": [_entry(model="qwen3.8-max")],
            "fallback_model": _entry(model="qwen3.8-max"),
        }
        chain = get_fallback_chain(cfg, primary_model="glm-5.3")
        assert len(chain) == 1
