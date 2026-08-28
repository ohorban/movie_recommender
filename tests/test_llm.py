"""The Claude client against SDKs with different `messages.create` signatures.

anthropic 1.x removed `temperature` (and `top_p`) from Messages.create. The
first attempt at handling that added a kwargs filter but never routed the call
sites through it — and the unit test only exercised the filter in isolation, so
it passed while every real call still raised. These tests drive `structured()`
and `text()` end to end against stand-in SDKs, which is the only way to catch
that class of mistake.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from movierec.config import Config
from movierec.enrich.llm import ClaudeClient


class _Block:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.input = payload


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Usage:
    input_tokens = 11
    output_tokens = 22


class _Response:
    def __init__(self, content: list[Any]) -> None:
        self.content = content
        self.usage = _Usage()


class ModernMessages:
    """anthropic 1.x: no `temperature`, no `top_p`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, *, model, max_tokens, system, messages, tools=None, tool_choice=None):
        self.calls.append({"model": model, "max_tokens": max_tokens, "tools": tools})
        if tools:
            return _Response([_Block({"ok": True, "model": model})])
        return _Response([_TextBlock("a sentence")])


class LegacyMessages:
    """anthropic 0.x: `temperature` accepted."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(
        self, *, model, max_tokens, system, messages, temperature=None, tools=None, tool_choice=None
    ):
        self.calls.append({"temperature": temperature})
        if tools:
            return _Response([_Block({"ok": True})])
        return _Response([_TextBlock("a sentence")])


class FakeSDK:
    def __init__(self, messages) -> None:
        self.messages = messages


def make_client(tmp_path, messages) -> ClaudeClient:
    cfg = Config(
        root=tmp_path,
        anthropic_api_key="sk-test",
        llm_model="claude-sonnet-5",
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db" / "t.db",
    )
    cfg.ensure_dirs()
    return ClaudeClient(cfg, client=FakeSDK(messages))


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def _structured(client: ClaudeClient, **over: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "kind": "test",
        "system": "s",
        "user": "u",
        "schema": SCHEMA,
        "tool_name": "record",
        "tool_description": "d",
        "temperature": 0.4,
        "use_cache": False,
    }
    kwargs.update(over)
    return client.structured(**kwargs)


def test_structured_works_on_an_sdk_without_temperature(tmp_path):
    """The regression: this raised TypeError for every call in production."""
    messages = ModernMessages()
    client = make_client(tmp_path, messages)
    assert _structured(client) == {"ok": True, "model": "claude-sonnet-5"}
    assert len(messages.calls) == 1


def test_text_works_on_an_sdk_without_temperature(tmp_path):
    client = make_client(tmp_path, ModernMessages())
    assert client.text(system="s", user="u", temperature=0.4) == "a sentence"


def test_temperature_is_still_sent_when_the_sdk_accepts_it(tmp_path):
    messages = LegacyMessages()
    client = make_client(tmp_path, messages)
    _structured(client)
    assert messages.calls[0]["temperature"] == 0.4, "do not drop a supported argument"


def test_every_create_call_site_routes_through_the_filter(tmp_path):
    """Guards the actual defect: a filter that exists but is not used.

    Both call sites must unpack `_call_kwargs`, otherwise an unsupported
    argument reaches the SDK regardless of what the filter says.
    """
    source = inspect.getsource(ClaudeClient)
    for fragment in source.split("self.client.messages.create(")[1:]:
        assert fragment.lstrip().startswith("**self._call_kwargs("), (
            "a messages.create() call bypasses the kwargs filter"
        )


def test_unsupported_arguments_are_dropped_not_renamed(tmp_path):
    client = make_client(tmp_path, ModernMessages())
    kwargs = client._call_kwargs(model="m", max_tokens=1, temperature=0.5, top_p=0.9, system="s")
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert kwargs["model"] == "m" and kwargs["system"] == "s"


def test_results_are_cached_by_content(tmp_path):
    messages = ModernMessages()
    client = make_client(tmp_path, messages)
    first = _structured(client, use_cache=True)
    second = _structured(client, use_cache=True)
    assert first == second
    assert len(messages.calls) == 1, "the second call should come from the cache"
    assert client.cache_hits == 1


def test_map_structured_preserves_order_and_survives_failures(tmp_path):
    class Flaky(ModernMessages):
        def create(self, *, model, max_tokens, system, messages, tools=None, tool_choice=None):
            if "boom" in str(messages):
                raise RuntimeError("upstream failure")
            return super().create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )

    client = make_client(tmp_path, Flaky())
    jobs = [
        {
            "kind": "test",
            "system": "s",
            "user": u,
            "schema": SCHEMA,
            "tool_name": "record",
            "tool_description": "d",
            "use_cache": False,
        }
        for u in ("first", "boom", "third")
    ]
    out = client.map_structured(jobs)
    assert len(out) == 3
    assert out[1] is None, "one bad job must not take the batch down"
    assert out[0] and out[2]


def test_usage_is_tracked(tmp_path):
    client = make_client(tmp_path, ModernMessages())
    _structured(client)
    usage = client.usage_summary()
    assert usage["calls"] == 1
    assert usage["input_tokens"] == 11 and usage["output_tokens"] == 22


def test_missing_api_key_is_a_clear_error(tmp_path):
    from movierec.enrich.llm import LLMError

    cfg = Config(root=tmp_path, anthropic_api_key="", data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        ClaudeClient(cfg, client=FakeSDK(ModernMessages()))


def test_a_kwargs_signature_disables_filtering(tmp_path):
    """A `**kwargs`-style create accepts anything; filtering would strip it all."""

    class Wrapped(ModernMessages):
        def create(self, **kw):
            return super().create(**kw)

    client = make_client(tmp_path, Wrapped())
    assert client._create_params == set(), "nothing to filter against"
    kwargs = client._call_kwargs(model="m", max_tokens=1, temperature=0.5)
    assert kwargs["temperature"] == 0.5, "pass everything through untouched"


def test_config_anchors_relative_paths_to_root(tmp_path):
    """A directly constructed Config must not resolve paths against the CWD.

    This bit for real: a test wrote its cache into the maintainer's live
    `data/cache/` because the default `data_dir` was the relative `Path("data")`.
    """
    cfg = Config(root=tmp_path)
    assert cfg.data_dir.is_absolute()
    assert cfg.db_path.is_absolute()
    assert cfg.data_dir == (tmp_path / "data").resolve()
    assert cfg.cache_dir == (tmp_path / "data" / "cache").resolve()


def test_config_leaves_absolute_paths_alone(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    cfg = Config(root=tmp_path, data_dir=elsewhere)
    assert cfg.data_dir == elsewhere
