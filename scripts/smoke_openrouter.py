from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.llm import parse_dotenv_value


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), parse_dotenv_value(value))


def main() -> int:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY missing")
        return 1
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_SMOKE_MODEL", "openai/gpt-4.1-mini")
    print(f"OPENROUTER_API_KEY set: prefix={key[:8]}..., length={len(key)}")
    print(f"base_url={base_url}")
    print(f"model={model}")

    from openai import OpenAI

    client = OpenAI(
        api_key=key,
        base_url=base_url,
        timeout=30.0,
    )
    try:
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": "Reply with exactly: ok"}],
            max_output_tokens=16,
        )
    except Exception as exc:
        print(f"OpenRouter Responses API smoke failed: {type(exc).__name__}: {exc}")
        return 2

    print("OpenRouter Responses API smoke succeeded")
    print(f"output_text={getattr(response, 'output_text', '')!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
