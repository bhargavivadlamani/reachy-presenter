# reachy-presenter

Reachy Mini robot presenter powered by Google ADK and Gemini Bidi streaming.

## What it does

- Runs as a persistent conversational agent — always listening, never locked into a fixed workflow
- User can say "load my deck" → agent parses PDF/PPTX and generates scripts via vision LLM
- User can say "present slide 2" → agent calls `present_slide`, performs robot gesture, reads script aloud
- Handles free-form Q&A and conversation between slides; calls `rag_query` to ground answers in ingested documents
- Native bidi audio: no separate TTS/STT, built-in interruption handling

## Running

**Simulator (audio via PC mic + speakers):**
```
python -m app.agent --audio
```
Connects to the Reachy Mini simulator (`localhost:8000`) for robot gestures. Sounddevice handles mic/speaker.

**Text CLI (no robot required):**
```
python -m app.agent
```

**Real robot:**
```python
from app.agent import run_for_robot, set_mini
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    set_mini(mini)
    run_for_robot(mini)
```

## Project layout

```
app/
  agent.py                  # Entry point: ADK agent, bidi streaming loop, audio I/O
  system_prompt.md          # Agent persona and tool instructions (loaded at runtime)
  audio_helpers/
    __init__.py             # MIC_SR, GEM_SR, VAD_THRESHOLD, to_pcm_bytes(), decode_pcm()
  tools/
    present_slide.py        # ADK tool: slide transition gesture + signal to read script aloud
    load_presentation.py    # ADK tool: parse PDF/PPTX, generate scripts via vision LLM
    generate_script.py      # Helper re-export (not an agent tool — takes PIL image)
  llm/
    gemini_client.py        # generate_script(), classify_slide() via FAU TRUSSED proxy (GPT-4o)
  parsers/
    pdf_parser.py           # PDF -> PIL Images via pdf2image (150 DPI)
    pptx_parser.py          # PPTX -> PIL Images via LibreOffice -> pdf2image
  robot/
    gestures.py             # slide_transition(), emotion_gesture() via reachy_mini SDK
```

## Environment

`.env` file needs:
```
GEMINI_API_KEY=...           # Gemini Bidi streaming (agent audio)
OPENAI_API_KEY=...           # FAU TRUSSED proxy — GPT-4o vision for script generation
QDRANT_URL=...               # Qdrant instance for RAG (e.g. http://localhost:6333)
RAG_EMBED_PROVIDER=ollama    # embedding provider used at ingest time
RAG_EMBED_MODEL=nomic-embed-text
```

Robot: `reachy-mini.local`, user `pollen`, project path `~/reachy-presenter/`

## Tech stack

- `google-adk` — `Agent`, `Runner`, `LiveRequestQueue`, `StreamingMode.BIDI`
- `google-genai` — Gemini Bidi (`gemini-2.5-flash-native-audio-preview-12-2025`) for live audio
- `openai` — FAU TRUSSED proxy (GPT-4o) for slide vision / script generation
- `sounddevice` — PC mic/speaker I/O, callback-based, `blocksize=512`, `latency=low`
- `scipy` — resample audio between hardware SR and Gemini's 16kHz in / 24kHz out
- `reachy-mini` — robot SDK
- `pdf2image` + `pdfplumber` — PDF parsing
- `python-pptx` + LibreOffice — PPTX parsing

## Agent architecture

```
run_audio_conversation()
  ├── sd.InputStream  (16kHz, blocksize=512, latency=low)
  ├── sd.OutputStream (24kHz, blocksize=512, latency=low, callback-based)
  └── _run_bidi_async()
        ├── upstream():    mic frames -> to_pcm_bytes() -> send_realtime() -> Gemini
        ├── downstream():  Gemini events -> decode_pcm() -> push_audio()
        │                               -> interrupted  -> clear_audio()
        │                               -> turn_complete -> unmute_audio() + idle timer
        └── idle timer:    30s after last turn_complete -> session closes

Interruption flow:
  mic callback: speech detected while bot playing (RMS > VAD_THRESHOLD)
    -> muted=True, interrupted=True  [local, ~21ms, no network round-trip]
  turn_complete from server:
    -> _unmute(): if interrupted flag set -> flush queue + stream abort/start + unmute
  event.interrupted from server (belt and suspenders):
    -> _clear(): same flush
```

## Agent tools

| Tool | Description |
|------|-------------|
| `load_presentation(file_path)` | Parse PDF/PPTX, generate all scripts, store for slide_number lookup |
| `present_slide(slide_number, script, document_text)` | Robot gesture + signal agent to read script aloud |
| `rag_query(query, collection_name)` | Retrieve relevant chunks from Qdrant; agent synthesizes the answer |

### Agentic RAG

`rag_query` enables the agent to answer student questions grounded in ingested documents (slides, textbooks, papers). It runs hybrid search (dense + sparse BM25) with cross-encoder reranking and returns formatted chunks with source citations. The main Gemini agent synthesizes the final answer.

**Setup:** pre-ingest documents before starting the agent:
```
python -m app.rag.ingest lecture.pdf --collection lecture
```
Collection name convention: lowercase filename stem. When `load_presentation("lecture.pdf")` is called, `rag_query` automatically targets the `lecture` collection — no extra config needed.

**Environment variables** (add to `.env`):
```
QDRANT_URL=http://localhost:6333
RAG_EMBED_PROVIDER=ollama          # must match what was used at ingest time
RAG_EMBED_MODEL=nomic-embed-text   # must match ingest embedding model
```

## Installing on robot

```bash
ssh pollen@reachy-mini.local
cd ~/reachy-presenter
pip install google-adk google-genai openai sounddevice scipy

# PPTX support
sudo apt-get install libreoffice poppler-utils
```

## Common issues

- **Interruption lag**: tune `VAD_THRESHOLD` in `audio_helpers/__init__.py` (default 0.02 RMS)
- **Bot doesn't respond / stays silent**: threshold too low — ambient noise is keeping output muted; raise it
- **No audio on simulator**: `media_backend="no_media"` — audio goes through sounddevice only
- **ADK bidi model**: must be `gemini-2.5-flash-native-audio-preview-12-2025`; standard flash models don't support `bidiGenerateContent`
- **Script generation slow**: FAU TRUSSED rate limit — `load_presentation` waits 7s between slides
- **sounddevice device**: if multiple audio devices, set `sd.default.device` before calling `run_audio_conversation()`
