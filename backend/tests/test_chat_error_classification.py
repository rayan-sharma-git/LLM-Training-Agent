"""Provider failures must be explained, not reduced to a generic sentence.

httpx's message for an HTTP error omits the response body, which is where the
provider says what is wrong (e.g. Ollama: "model 'llama3.2' not found"). The
classifier therefore reads the status and body, and the generic fallback names
the real reason.
"""
from __future__ import annotations


from chat.engine import ChatEngine, _first_sentence, _http_error_parts


class FakeResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text


class HttpError(Exception):
    def __init__(self, status_code: int, text: str):
        super().__init__(f"Client error '{status_code}' for url 'http://localhost:11434/api/chat'")
        self.response = FakeResponse(status_code, text)


def classify(error: Exception, provider: str = "ollama"):
    engine = ChatEngine(provider_name=provider)
    return engine._classify_provider_error(error)


def test_ollama_missing_model_is_explained_with_the_pull_command():
    err = classify(HttpError(404, '{"error":"model \'llama3.2\' not found, try pulling it first"}'))
    assert err.error_code == "INVALID_MODEL"
    assert "ollama pull" in err.message


def test_http_401_is_an_invalid_key_even_without_keywords_in_the_text():
    err = classify(HttpError(401, "{}"), provider="openai")
    assert err.error_code == "INVALID_API_KEY"


def test_unknown_failure_names_the_provider_reason():
    err = classify(HttpError(500, '{"error":"llama runner process has terminated: out of memory"}'))
    assert err.error_code == "PROVIDER_UNAVAILABLE"
    assert "out of memory" in err.message


def test_secrets_in_the_provider_body_never_reach_the_message():
    err = classify(HttpError(500, '{"error":"bad request using key sk-abcdefghijklmnopqrstuvwxyz123456"}'), provider="openai")
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in err.message
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in str(err.details)


def test_plain_exceptions_without_a_response_still_classify():
    assert classify(ConnectionError("connection refused")).error_code == "PROVIDER_UNAVAILABLE"
    assert classify(TimeoutError("timed out")).error_code == "PROVIDER_TIMEOUT"


def test_helpers():
    assert _http_error_parts(ValueError("x")) == (None, "")
    assert _http_error_parts(HttpError(404, "body")) == (404, "body")
    assert _first_sentence('{"error":{"message":"nope"}}') == "nope"
    assert _first_sentence("a\n  b") == "a b"
