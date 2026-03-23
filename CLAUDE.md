# Reachy Mini Presenter — Project Context

## What This Project Does
Reachy Mini robot autonomously presents PDF/PPTX slide decks. It:
1. Parses slides as images (vision-based, not text extraction)
2. Sends each slide image to Gemini Flash to generate a spoken script + classify emotion
3. Uses Gemini Live API for TTS, VAD, echo cancellation, and live Q&A natively
4. Moves expressively: slide transitions, emotion gestures, face tracking during speech

Run: `python -m app.main path/to/slides.pdf`

## Project Layout
```
app/
  main.py                  # Entry point: parses slides, generates scripts, runs presentation loop
  llm/
    gemini_client.py       # Gemini Flash: generate_script(), classify_slide(), transcribe_audio(), answer_question()
  parsers/
    pdf_parser.py          # PDF → PIL Images via pdf2image (150 DPI)
    pptx_parser.py         # PPTX → PIL Images via LibreOffice → pdf2image
  robot/
    live_presenter.py      # ACTIVE: Gemini Live API — TTS + mic VAD + Q&A in one streaming session
    gestures.py            # slide_transition() and emotion_gesture() using reachy_mini SDK
    vision.py              # MediaPipe face tracking thread (track_faces_during_speech)
    tts.py                 # DEAD CODE — replaced by live_presenter.py
    qa_handler.py          # DEAD CODE — replaced by live_presenter.py (Gemini Live handles Q&A natively)
  llm/
    openai_client.py       # DEAD CODE — fully replaced by gemini_client.py
```

## Environment
- `.env` file needs `GEMINI_API_KEY=...`
- Robot hostname: `reachy-mini.local`, user: `pollen`
- Robot project path: `~/reachy-presenter/`
- Run via: `cd ~/reachy-presenter && python -m app.main slides.pdf`

## Tech Stack
- `google-genai` — Gemini Flash (`gemini-2.0-flash`) for vision/script/classify; Gemini Live (`gemini-2.0-flash-live-001`) for speech
- `sounddevice` — captures mic audio (16kHz 16-bit PCM) to stream to Gemini Live
- `mediapipe` — BlazeFace face detection for audience eye contact (model_selection=0, short-range, avoids aarch64 protobuf bug)
- `scipy` — resample Gemini's 24kHz output to robot's native sample rate
- `reachy-mini` — robot SDK: `goto_target`, `set_target`, `look_at_image`, `media.push_audio_sample`, `media.get_frame`
- `pdf2image` + `pdfplumber` — PDF parsing
- `python-pptx` + LibreOffice — PPTX parsing (LibreOffice must be installed: `sudo apt-get install libreoffice poppler-utils`)

## Architecture: Per-Slide Flow
```
1. slide_transition(mini)        ~1.2s  glance down → look up (checking notes)
2. emotion_gesture(mini, tone)   ~0.6–1.1s  expressive move based on slide tone
3. present_slide(mini, script, document_text)
   ├── Gemini Live session opens
   ├── Script text sent → Gemini speaks it (TTS via AUDIO modality, voice "Aoede")
   ├── Thread A: mic_reader() → streams 16kHz PCM chunks to Gemini Live (VAD + AEC built-in)
   ├── Thread B: receive_audio() → plays Gemini's 24kHz output through robot speaker
   ├── Thread C: track_faces_during_speech() → MediaPipe face detection at 4Hz, head tracking
   ├── Audience can interrupt at any time — Gemini Live handles it natively
   └── On turn_complete → wait 2s for follow-up → done.set() → advance to next slide
```

## Key Implementation Details

### live_presenter.py
- Uses `asyncio.run(_run(...))` per slide — blocks until slide + Q&A complete
- Gemini Live requires `api_version="v1beta"` (different from `v1` used by generate_content)
- `done = asyncio.Event()` — set when `server_content.turn_complete` received
- `mic_queue` bridges sync sounddevice thread → async Gemini send coroutine via `loop.call_soon_threadsafe`
- Audio format in: `audio/pcm;rate=16000` (16kHz int16 PCM)
- Audio format out: 24kHz int16 PCM → float32 → scipy_resample → `mini.media.push_audio_sample`

### vision.py
- `track_faces_during_speech(mini, stop_event: threading.Event)` — stops when event is set
- `model_selection=0` (short-range) — model_selection=1 crashes on aarch64 Linux (protobuf bug)
- BGR→RGB: `frame[:, :, ::-1].copy()` — no cv2 needed, must be C-contiguous for MediaPipe
- Fallback when no face: slow sinusoidal yaw scan via `mini.set_target(head=create_head_pose(yaw=...))`

### gemini_client.py
- Two separate clients: `api_version="v1"` for generate_content, `api_version="v1beta"` for Live
- `generate_script(PIL_image)` → spoken script for slide
- `classify_slide(PIL_image)` → one of: excited / neutral / questioning / serious
- `transcribe_audio(wav_bytes)` → Gemini audio transcription (fallback, Live API handles this natively now)
- `answer_question(question, document_text)` → 2-3 sentence spoken answer (fallback, Live handles this)
- Rate limit: 2-second sleep between calls to stay within free tier (15 req/min)

### gestures.py
- `slide_transition`: pitch -20° (minjerk 0.5s) → neutral (ease_in_out 0.7s)
- `emotion_gesture("excited")`: antennas 40° + pitch 8° (cartoon) → neutral
- `emotion_gesture("questioning")`: roll 12° → neutral
- `emotion_gesture("serious")`: pitch -5° + antennas -15°
- `emotion_gesture("neutral")`: return to neutral

## Installing on Robot
```bash
# SSH into robot
ssh pollen@reachy-mini.local

# In project venv
cd ~/reachy-presenter
pip install google-genai sounddevice mediapipe scipy

# For PPTX support
sudo apt-get install libreoffice poppler-utils

# Uninstall old stuff (no longer needed)
pip uninstall openai edge-tts pynput -y
```

## Deploying Changes from Mac
```bash
# From Desktop/Reachy-Presenter/reachy-presenter/
scp app/robot/live_presenter.py pollen@reachy-mini.local:~/reachy-presenter/app/robot/live_presenter.py
scp app/robot/vision.py         pollen@reachy-mini.local:~/reachy-presenter/app/robot/vision.py
scp app/robot/gestures.py       pollen@reachy-mini.local:~/reachy-presenter/app/robot/gestures.py
scp app/main.py                 pollen@reachy-mini.local:~/reachy-presenter/app/main.py
scp app/llm/gemini_client.py    pollen@reachy-mini.local:~/reachy-presenter/app/llm/gemini_client.py
scp requirements.txt            pollen@reachy-mini.local:~/reachy-presenter/requirements.txt
```

## Common Issues & Fixes
- **MediaPipe crash on aarch64**: use `model_selection=0`, NOT 1 (full-range model has protobuf parse error on ARM)
- **Gemini Live needs v1beta**: `http_options={"api_version": "v1beta"}` in the client for live_presenter.py
- **generate_content uses v1**: separate `genai.Client` with `api_version="v1"` in gemini_client.py
- **sounddevice mic**: if robot has multiple audio devices, may need to set `sd.default.device`
- **scp new file fails**: the directory must already exist on robot. Check with `ssh pollen@reachy-mini.local "ls ~/reachy-presenter/app/robot/"`
- **Rate limit**: free Gemini tier is 15 req/min — `time.sleep(2)` between calls in main.py pre-generation loop

## What's Dead Code (don't touch, don't use)
- `app/robot/tts.py` — old edge-tts based TTS, superseded by Gemini Live
- `app/robot/qa_handler.py` — old sounddevice VAD + Gemini transcription Q&A, superseded by Gemini Live
- `app/llm/openai_client.py` — old OpenAI GPT-4o calls, fully replaced by gemini_client.py
