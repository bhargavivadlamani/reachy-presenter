# Reachy Mini Presenter — Project Context

## What This Project Does
Reachy Mini robot agent powered by Google ADK + Gemini Bidi streaming. It:
1. Runs as a persistent conversational agent — always listening, never locked into a fixed workflow
2. `load_presentation(file_path)` tool: parses PDF/PPTX, generates spoken scripts via GPT-4o vision
3. `present_slide(slide_number)` tool: performs robot gesture, signals agent to read script via bidi TTS
4. Native bidi audio (Gemini Live): no separate TTS/STT, built-in VAD + interruption handling

Run agent (simulator): `python -m app.agent --audio`
Run agent (text CLI):   `python -m app.agent`
Real robot: call `set_mini(mini)` then `run_for_robot(mini)` from `app.agent`

## Project Layout
```
app/
  agent.py                  # ACTIVE: ADK agent, bidi streaming loop, run_for_robot(), run_audio_conversation()
  system_prompt.md          # Agent persona + tool instructions (loaded at runtime)
  audio_helpers/
    __init__.py             # MIC_SR=16000, GEM_SR=24000, VAD_THRESHOLD=0.02, to_pcm_bytes(), decode_pcm()
  tools/
    present_slide.py        # ADK tool: slide_transition gesture + return script for agent TTS
                            # Also owns _mini_ref + set_mini() — call set_mini() before audio session
    load_presentation.py    # ADK tool: parse file, generate scripts, store in module state
    generate_script.py      # Helper re-export from gemini_client (not an ADK tool — takes PIL image)
  llm/
    gemini_client.py        # generate_script(PIL_image), classify_slide(PIL_image) via FAU TRUSSED/GPT-4o
  parsers/
    pdf_parser.py           # PDF → PIL Images via pdf2image (150 DPI)
    pptx_parser.py          # PPTX → PIL Images via LibreOffice → pdf2image
  robot/
    gestures.py             # slide_transition(), emotion_gesture() via reachy_mini SDK
```

## Environment
- `.env` needs `GEMINI_API_KEY=...` (Gemini bidi) and `OPENAI_API_KEY=...` (FAU TRUSSED proxy)
- Robot hostname: `reachy-mini.local`, user: `pollen`, project path: `~/reachy-presenter/`

## Tech Stack
- `google-adk` — `Agent`, `Runner`, `InMemorySessionService`, `LiveRequestQueue`, `StreamingMode.BIDI`
- `google-genai` — Gemini Bidi (`gemini-2.5-flash-native-audio-preview-12-2025`) for live audio
- `openai` — FAU TRUSSED proxy (GPT-4o vision) for script generation
- `sounddevice` — PC mic/speaker I/O, callback-based (`blocksize=512`, `latency=low`)
- `scipy` — resample audio between hardware SR and Gemini's 16kHz in / 24kHz out
- `reachy-mini` — robot SDK: `goto_target`, `set_target`, `media.push_audio_sample`
- `pdf2image` + `pdfplumber` — PDF parsing
- `python-pptx` + LibreOffice — PPTX parsing

## Architecture: agent.py

### Modes
- `python -m app.agent --audio`: connects to simulator (`localhost_only`, `no_media`), sounddevice for audio
- `python -m app.agent`: text CLI via `run_async`, no robot needed
- `run_for_robot(mini)`: real robot — owns media lifecycle (start/stop playing+recording)

### Bidi streaming loop (`_run_bidi_async`)
```
upstream():   mic frames -> to_pcm_bytes() -> send_realtime() -> Gemini Live
downstream(): Gemini events -> inline_data -> decode_pcm() -> push_audio()
                            -> interrupted -> clear_audio()
                            -> turn_complete -> unmute_audio() + idle timer reset
idle timer:   30s after last turn_complete -> done.set() -> session closes
```

### Interruption handling (sounddevice mode)
```
mic callback: speech detected while bot playing (RMS > VAD_THRESHOLD)
  -> muted=True, interrupted=True  [local, immediate, no network round-trip]
turn_complete from server:
  -> _unmute(): if interrupted flag -> flush queue + out_stream.abort/start + unmute
event.interrupted from server (belt and suspenders):
  -> _clear(): same flush
```
Key: `_push()` never touches `muted`. Only `_flush_and_unmute()` unmutes — prevents old queued audio replaying after interruption.

### Audio constants (app/audio_helpers/__init__.py)
- `VAD_THRESHOLD = 0.02` — RMS amplitude for local speech detection. Raise if ambient noise triggers it.
- `MIC_SR = 16000`, `GEM_SR = 24000` — Gemini Live wire format

### Tools
- `present_slide(slide_number, script, document_text)` — gesture + return script for agent to speak
  - `_mini_ref` lives in `app/tools/present_slide.py`; call `set_mini(mini)` before session starts
  - `slide_number` looks up pre-generated script from `load_presentation` state
- `load_presentation(file_path)` — parses file, calls `generate_script` per slide (7s delay between for rate limit), stores scripts + document_text in module state

### gemini_client.py
- Uses FAU TRUSSED OpenAI-compatible proxy (`https://fauengtrussed.fau.edu/provider/generic`)
- `generate_script(PIL_image)` → 4-6 sentence spoken script
- `classify_slide(PIL_image)` → excited / neutral / questioning / serious

### gestures.py
- `slide_transition`: pitch -20° (minjerk 0.5s) → neutral (ease_in_out 0.7s)
- `emotion_gesture("excited")`: antennas 40° + pitch 8° → neutral
- `emotion_gesture("questioning")`: roll 12° → neutral
- `emotion_gesture("serious")`: pitch -5° + antennas -15°

## Installing on Robot
```bash
ssh pollen@reachy-mini.local
cd ~/reachy-presenter
pip install google-adk google-genai openai sounddevice scipy

# PPTX support
sudo apt-get install libreoffice poppler-utils
```

## Deploying Changes
```bash
scp app/agent.py                         pollen@reachy-mini.local:~/reachy-presenter/app/agent.py
scp app/system_prompt.md                 pollen@reachy-mini.local:~/reachy-presenter/app/system_prompt.md
scp app/audio_helpers/__init__.py        pollen@reachy-mini.local:~/reachy-presenter/app/audio_helpers/__init__.py
scp app/tools/present_slide.py           pollen@reachy-mini.local:~/reachy-presenter/app/tools/present_slide.py
scp app/tools/load_presentation.py       pollen@reachy-mini.local:~/reachy-presenter/app/tools/load_presentation.py
scp app/robot/gestures.py               pollen@reachy-mini.local:~/reachy-presenter/app/robot/gestures.py
scp app/llm/gemini_client.py            pollen@reachy-mini.local:~/reachy-presenter/app/llm/gemini_client.py
scp requirements.txt                     pollen@reachy-mini.local:~/reachy-presenter/requirements.txt
```

## Common Issues & Fixes
- **Bot stays silent / doesn't respond**: `VAD_THRESHOLD` too low — ambient noise muting output; raise it in `audio_helpers/__init__.py`
- **Interruption lag**: same threshold — lower = faster cut but triggers on noise
- **No audio on simulator**: `media_backend="no_media"` — audio is sounddevice only, not GStreamer
- **ADK bidi model**: must use `gemini-2.5-flash-native-audio-preview-12-2025`; regular models don't support `bidiGenerateContent`
- **Script generation rate limit**: FAU TRUSSED ~10 req/min — `load_presentation` waits 7s between slides
- **sounddevice device**: if multiple audio devices, set `sd.default.device` before `run_audio_conversation()`
- **MediaPipe (if re-added)**: use `model_selection=0` on aarch64 — model_selection=1 crashes (protobuf bug)
