"""Cohere v2 requests must use the v2 `messages` shape (v1's message/chat_history is rejected)."""
from __future__ import annotations

from ai.providers.cohere_provider import CohereProvider


def _provider() -> CohereProvider:
    return CohereProvider(api_key="test-key", model="command-r")


def test_payload_uses_v2_messages_and_keeps_the_system_prompt():
    payload = _provider().build_payload(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "q"}],
        system="be brief",
        temperature=0.2,
    )
    assert "message" not in payload and "chat_history" not in payload
    assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant", "user"]
    assert payload["messages"][0]["content"] == "be brief"
    assert payload["messages"][-1]["content"] == "q"
    assert payload["model"] == "command-r" and payload["temperature"] == 0.2


def test_existing_system_message_is_not_duplicated():
    payload = _provider().build_payload(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], system="other"
    )
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
