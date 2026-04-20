"""
Quick API key / model availability tester.
Run: python test_keys.py
"""
import os, sys, textwrap
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
# TRUSSED portal issues one key per project — try dedicated var, else reuse GEMINI_API_KEY
TRUSSED_KEY = (os.environ.get("TRUSSED_API_KEY")
               or os.environ.get("GEMINI_API_KEY")
               or "").strip()
TRUSSED_BASE = "https://fauengtrussed.fau.edu/provider/generic"

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m-\033[0m"

def section(title):
    print(f"\n{'─'*50}\n  {title}\n{'─'*50}")

# ── FAU TRUSSED proxy ─────────────────────────────────

TRUSSED_CHAT_URL  = "https://fauengtrussed.fau.edu/provider/generic/chat/completions"
TRUSSED_EMBED_URL = "https://fauengtrussed.fau.edu/provider/generic/embeddings"

def _trussed_headers(key: str) -> dict:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

def _test_trussed_chat(model: str, key: str = None):
    import requests as req
    k = key or TRUSSED_KEY
    try:
        r = req.post(
            TRUSSED_CHAT_URL,
            headers=_trussed_headers(k),
            json={"model": model,
                  "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                  "max_tokens": 10},
            timeout=20,
        )
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            print(f"  {PASS}  {model:50s}  → {text[:60]!r}")
        else:
            err = textwrap.shorten(r.text, 80)
            print(f"  {FAIL}  {model:50s}  → HTTP {r.status_code}: {err}")
    except Exception as e:
        print(f"  {FAIL}  {model:50s}  → {textwrap.shorten(str(e), 80)}")


def _test_trussed_embedding(model: str, key: str = None):
    import requests as req
    k = key or TRUSSED_KEY
    try:
        r = req.post(
            TRUSSED_EMBED_URL,
            headers=_trussed_headers(k),
            json={"model": model, "input": "Hello world"},
            timeout=20,
        )
        if r.status_code == 200:
            vec = r.json()["data"][0]["embedding"]
            print(f"  {PASS}  {model:50s}  → dim={len(vec)}")
        else:
            err = textwrap.shorten(r.text, 80)
            print(f"  {FAIL}  {model:50s}  → HTTP {r.status_code}: {err}")
    except Exception as e:
        print(f"  {FAIL}  {model:50s}  → {textwrap.shorten(str(e), 80)}")


# ── Gemini ────────────────────────────────────────────

def test_gemini_text(model: str):
    """Gemini generate_content — text in, text out."""
    try:
        from google import genai
        client = genai.Client(
            api_key=GEMINI_KEY,
            http_options={"api_version": "v1"}
        )
        resp = client.models.generate_content(
            model=model,
            contents="Reply with exactly: OK"
        )
        text = resp.text.strip()
        ok = "OK" in text or len(text) > 0
        print(f"  {PASS if ok else FAIL}  {model:50s}  → {text[:60]!r}")
    except Exception as e:
        short = textwrap.shorten(str(e), 80)
        print(f"  {FAIL}  {model:50s}  → {short}")


def test_gemini_embedding(model: str):
    """Gemini embedContent."""
    try:
        from google import genai
        client = genai.Client(
            api_key=GEMINI_KEY,
            http_options={"api_version": "v1"}
        )
        resp = client.models.embed_content(
            model=model,
            contents="Hello world"
        )
        vec = resp.embeddings[0].values
        ok = len(vec) > 0
        print(f"  {PASS if ok else FAIL}  {model:50s}  → dim={len(vec)}")
    except Exception as e:
        short = textwrap.shorten(str(e), 80)
        print(f"  {FAIL}  {model:50s}  → {short}")


def test_gemini_live(model: str):
    """Gemini Live — just opens a session and immediately closes it."""
    try:
        import asyncio
        from google import genai
        from google.genai import types

        async def _ping():
            client = genai.Client(
                api_key=GEMINI_KEY,
                http_options={"api_version": "v1beta"}
            )
            config = types.LiveConnectConfig(
                response_modalities=["TEXT"],
                system_instruction="Reply with: LIVE_OK",
            )
            async with client.aio.live.connect(model=model, config=config) as session:
                await session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text="ping")])
                )
                async for resp in session.receive():
                    if resp.server_content and resp.server_content.turn_complete:
                        break
                    if resp.server_content and resp.server_content.model_turn:
                        for part in resp.server_content.model_turn.parts:
                            if part.text:
                                return part.text.strip()
            return "(no text)"

        text = asyncio.run(_ping())
        print(f"  {PASS}  {model:50s}  → {text[:60]!r}")
    except Exception as e:
        short = textwrap.shorten(str(e), 80)
        print(f"  {FAIL}  {model:50s}  → {short}")


# ── OpenAI ────────────────────────────────────────────

def test_openai_chat(model: str):
    """OpenAI chat completion."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        ok = len(text) > 0
        print(f"  {PASS if ok else FAIL}  {model:50s}  → {text[:60]!r}")
    except Exception as e:
        short = textwrap.shorten(str(e), 80)
        print(f"  {FAIL}  {model:50s}  → {short}")


def test_openai_embedding(model: str):
    """OpenAI embeddings."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        resp = client.embeddings.create(model=model, input="Hello world")
        vec = resp.data[0].embedding
        ok = len(vec) > 0
        print(f"  {PASS if ok else FAIL}  {model:50s}  → dim={len(vec)}")
    except Exception as e:
        short = textwrap.shorten(str(e), 80)
        print(f"  {FAIL}  {model:50s}  → {short}")


# ── Main ──────────────────────────────────────────────

def main():
    print("\n=== API Key / Model Test ===")
    print(f"  GEMINI_API_KEY : {'set (' + GEMINI_KEY[:8] + '…)' if GEMINI_KEY else 'NOT SET'}")
    print(f"  OPENAI_API_KEY : {'set (' + OPENAI_KEY[:8] + '…)' if OPENAI_KEY else 'NOT SET'}")
    print(f"  TRUSSED_API_KEY: {'set (' + TRUSSED_KEY[:8] + '…)' if TRUSSED_KEY else 'NOT SET'}")
    print(f"  TRUSSED_BASE   : {TRUSSED_BASE}")

    openai_trussed = OPENAI_KEY   # key labelled OPENAI in .env
    gemini_trussed = GEMINI_KEY   # key labelled GEMINI in .env

    section("FAU TRUSSED — OpenAI models  (using OPENAI_API_KEY)")
    if openai_trussed:
        for m in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]:
            _test_trussed_chat(m, key=openai_trussed)
        _test_trussed_embedding("text-embedding-3-small", key=openai_trussed)
    else:
        print(f"  {SKIP}  OPENAI_API_KEY not set")

    section("FAU TRUSSED — Gemini models  (using GEMINI_API_KEY)")
    if gemini_trussed:
        for m in ["gemini-2.0-flash", "gemini-2.5-flash",
                  "gemini-1.5-pro", "gemini-1.5-flash",
                  "google/gemini-2.0-flash", "google/gemini-1.5-pro"]:
            _test_trussed_chat(m, key=gemini_trussed)
        for m in ["gemini-embedding-2-preview", "text-embedding-004"]:
            _test_trussed_embedding(m, key=gemini_trussed)
    else:
        print(f"  {SKIP}  GEMINI_API_KEY not set")

    # Gemini text
    section("Gemini — text generation (v1)")
    if GEMINI_KEY:
        test_gemini_text("gemini-2.0-flash")
        test_gemini_text("gemini-2.5-flash")
    else:
        print(f"  {SKIP}  GEMINI_API_KEY not set — skipping")

    # Gemini embedding
    section("Gemini — embeddings (v1)")
    if GEMINI_KEY:
        test_gemini_embedding("gemini-embedding-2-preview")
    else:
        print(f"  {SKIP}  GEMINI_API_KEY not set — skipping")

    # Gemini Live
    section("Gemini — Live API (v1beta)")
    if GEMINI_KEY:
        test_gemini_live("gemini-2.5-flash-native-audio-preview-12-2025")
    else:
        print(f"  {SKIP}  GEMINI_API_KEY not set — skipping")

    # OpenAI chat
    section("OpenAI — chat completions")
    if OPENAI_KEY:
        test_openai_chat("gpt-4o")
        test_openai_chat("gpt-4o-mini")
        test_openai_chat("gpt-4.5-preview")   # nearest real model to "gpt-5.4"
    else:
        print(f"  {SKIP}  OPENAI_API_KEY not set — skipping")

    # OpenAI embedding
    section("OpenAI — embeddings")
    if OPENAI_KEY:
        test_openai_embedding("text-embedding-3-small")
    else:
        print(f"  {SKIP}  OPENAI_API_KEY not set — skipping")

    print()


if __name__ == "__main__":
    main()
