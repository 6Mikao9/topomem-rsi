"""Standard-library transport for the topology RSI model provider."""

import json
import os
from pathlib import Path
import urllib.error
import urllib.request
import uuid


DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_ENDPOINT = "https://opencode.ai/inference/go/openai/v1/chat/completions"


class APIError(Exception):
    """Safe public failure classification, with no response body or credentials."""


def load_api_key(key_file=None):
    """Read only the explicitly named file, or OPENCODE_GO_API_KEY."""
    if key_file:
        raw = Path(key_file).read_text(encoding="utf-8-sig").strip()
        if raw.startswith("{"):
            try:
                raw = json.loads(raw)["api_key"]
            except (ValueError, KeyError, TypeError):
                raise ValueError("key file JSON requires an api_key string") from None
    else:
        raw = os.environ.get("OPENCODE_GO_API_KEY", "").strip()
    if not isinstance(raw, str) or not raw or any(ch.isspace() for ch in raw):
        raise ValueError("provide a single API key via --key-file or OPENCODE_GO_API_KEY")
    return raw


def _consume_sse(response):
    """Collect final-content deltas and usage, discarding all reasoning deltas."""
    content, usage, lines = [], {}, []
    finish_reason, event_name, done = None, "", False

    def consume_event():
        nonlocal finish_reason, done
        if event_name == "error":
            raise APIError("stream reported an error")
        if not lines:
            return
        data = "\n".join(lines).strip()
        if data == "[DONE]":
            done = True
            return
        try:
            chunk = json.loads(data)
        except (ValueError, TypeError):
            raise APIError("invalid stream event") from None
        if not isinstance(chunk, dict):
            raise APIError("invalid stream event")
        if "error" in chunk:
            raise APIError("stream reported an error")
        if isinstance(chunk.get("usage"), dict):
            # Usage chunks contain cumulative totals, not per-chunk increments.
            usage.update(chunk["usage"])
        choices = chunk.get("choices", [])
        if not isinstance(choices, list):
            raise APIError("invalid stream event")
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise APIError("invalid stream event")
            final_text = delta.get("content")
            if final_text is not None:
                if not isinstance(final_text, str):
                    raise APIError("invalid stream event")
                content.append(final_text)
            # reasoning_content/reasoning/text fields are deliberately ignored.
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

    for raw_line in response:
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
        except (UnicodeError, AttributeError):
            raise APIError("invalid stream event") from None
        if not line:
            consume_event()
            lines, event_name = [], ""
            if done:
                break
        elif line.startswith(":"):
            continue  # SSE keepalive/comment; no model content.
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
    if not done and lines:
        consume_event()
    if not done:
        raise APIError("incomplete event stream")
    # A completed stream may spend its budget on reasoning and provide no final
    # content. Return its finish reason/usage so the caller can retain evidence
    # before the strict candidate parser rejects the empty proposal.
    return {"choices": [{"message": {"content": "".join(content)},
                         "finish_reason": finish_reason}], "usage": usage}


def make_requester(api_key, endpoint=DEFAULT_ENDPOINT, timeout=240):
    if not endpoint.startswith("https://"):
        raise ValueError("LLM endpoint must use HTTPS")
    if timeout <= 0:
        raise ValueError("request timeout must be positive")
    # One conversation/session per requester, stable across feedback rounds.
    # This transport header is deliberately excluded from persisted artifacts.
    session_id = uuid.uuid4().hex

    def request(payload):
        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json",
                     "User-Agent": "rsi-sparse-memory/0.1.0", "Accept": "application/json",
                     "x-opencode-session": session_id},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if payload.get("stream"):
                    return _consume_sse(response)
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise APIError("HTTP status " + str(exc.code)) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise APIError("connection or timeout failure") from None
        try:
            result = json.loads(body)
        except (ValueError, UnicodeError):
            raise APIError("API returned invalid JSON") from None
        if not isinstance(result, dict):
            raise APIError("API returned an unexpected response shape")
        return result

    request.endpoint = endpoint
    return request


def token_usage(response):
    usage = response.get("usage", {})
    allowed = {"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens",
               "output_tokens", "prompt_tokens_details", "completion_tokens_details",
               "cache_read_input_tokens", "cache_creation_input_tokens"}

    def numeric(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, dict):
            return {k: v for k, v in value.items() if isinstance(k, str) and k.replace("_", "").isalnum()
                    and len(k) < 65 and isinstance(v, (int, float)) and not isinstance(v, bool)}
        return None

    return {key: numeric(value) for key, value in usage.items() if key in allowed and numeric(value) is not None} if isinstance(usage, dict) else {}
