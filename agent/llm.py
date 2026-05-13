from __future__ import annotations

import os
from pathlib import Path
import time


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def make_openai_client(provider: str = "openai"):
    from openai import OpenAI

    load_dotenv()
    provider = provider.lower()
    if provider == "openai":
        return OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            timeout=90.0,
        )
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        default_headers=_openrouter_headers(),
        timeout=90.0,
    )


def create_response_with_retries(client, **kwargs):
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            return client.responses.create(**kwargs)
        except Exception as exc:  # Network/provider errors are common in long evals.
            last_error = exc
            if attempt == 4:
                break
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError("Responses API call failed after retries") from last_error


def _openrouter_headers() -> dict[str, str]:
    headers = {}
    if os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
    if os.getenv("OPENROUTER_APP_TITLE"):
        headers["X-Title"] = os.environ["OPENROUTER_APP_TITLE"]
    return headers
